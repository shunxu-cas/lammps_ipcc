/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 This software is distributed under the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 Contributing author: Shun Xu (CAS), W. Michael Brown (Intel)
 ------------------------------------------------------------------------- */

#include <math.h>
#include "pair_dpd_intel.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"

#include "suffix.h"
using namespace LAMMPS_NS;

#define FC_PACKED1_T typename ForceConst<flt_t>::fc_packed1
#define FC_PACKED2_T typename ForceConst<flt_t>::fc_packed2

#define EPSILON 1.0e-10
/* ---------------------------------------------------------------------- */
#ifdef _LMP_INTEL_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif

double RanMarsOffload_uniform(RanMarsOffload *rng) { //uniform RN
    double uni = rng->u[rng->i97] - rng->u[rng->j97];
    if (uni < 0.0)
        uni += 1.0;
    rng->u[rng->i97] = uni;
    rng->i97--;
    if (rng->i97 == 0)
        rng->i97 = 97;
    rng->j97--;
    if (rng->j97 == 0)
        rng->j97 = 97;
    rng->c -= rng->cd;
    if (rng->c < 0.0)
        rng->c += rng->cm;
    uni -= rng->c;
    if (uni < 0.0)
        uni += 1.0;
    return uni;
}
void RanMarsOffload_init(RanMarsOffload *rng, int seed) {
    int ij, kl, i, j, k, l, ii, jj, m;
    double s, t;

    if (seed <= 0 || seed > 900000000) {
        seed = 100;
        //"Invalid seed for Marsaglia random # generator";
    }
    rng->save = 0;
    //u = new double[97 + 1];

    ij = (seed - 1) / 30082;
    kl = (seed - 1) - 30082 * ij;
    i = (ij / 177) % 177 + 2;
    j = ij % 177 + 2;
    k = (kl / 169) % 178 + 1;
    l = kl % 169;
    for (ii = 1; ii <= 97; ii++) {
        s = 0.0;
        t = 0.5;
        for (jj = 1; jj <= 24; jj++) {
            m = ((i * j) % 179) * k % 179;
            i = j;
            j = k;
            k = m;
            l = (53 * l + 1) % 169;
            if ((l * m) % 64 >= 32)
                s = s + t;
            t = 0.5 * t;
        }
        rng->u[ii] = s;
    }
    rng->c = 362436.0 / 16777216.0;
    rng->cd = 7654321.0 / 16777216.0;
    rng->cm = 16777213.0 / 16777216.0;
    rng->i97 = 97;
    rng->j97 = 33;
    RanMarsOffload_uniform(rng);
}
double RanMarsOffload_gaussian(RanMarsOffload *rng) { //uniform RN
    double first, v1, v2, rsq, fac;

    if (!rng->save) {
        int again = 1;
        while (again) {
            v1 = 2.0 * RanMarsOffload_uniform(rng) - 1.0;
            v2 = 2.0 * RanMarsOffload_uniform(rng) - 1.0;
            rsq = v1 * v1 + v2 * v2;
            if (rsq < 1.0 && rsq != 0.0)
                again = 0;
        }
        fac = sqrt(-2.0 * log(rsq) / rsq);
        rng->second = v1 * fac;
        first = v2 * fac;
        rng->save = 1;
    } else {
        first = rng->second;
        rng->save = 0;
    }
    return first;
}
#ifdef _LMP_INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

PairDPDIntel::PairDPDIntel(LAMMPS *lmp) :
        PairDPD(lmp) {
    suffix_flag |= Suffix::INTEL;
    respa_enable = 0;
    random_thr = NULL;
    fix = NULL;
    off_threads = -1;
}

PairDPDIntel::~PairDPDIntel() {
#ifdef _LMP_INTEL_OFFLOAD
    RanMarsOffload * random_thr=this->random_thr;
    int othreads=this->off_threads;
#pragma offload_transfer target(mic:_cop) if(othreads>0) \
    nocopy(random_thr:length(othreads) alloc_if(0) free_if(1))
#endif

    memory->destroy(random_thr);
}

/* ---------------------------------------------------------------------- */

void PairDPDIntel::compute(int eflag, int vflag) {
    if (fix->precision() == FixIntel::PREC_MODE_MIXED)
        compute<float, double>(eflag, vflag, fix->get_mixed_buffers(),
                force_const_single);
    else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE)
        compute<double, double>(eflag, vflag, fix->get_double_buffers(),
                force_const_double);
    else
        compute<float, float>(eflag, vflag, fix->get_single_buffers(),
                force_const_single);

    fix->balance_stamp();
    vflag_fdotr = 0;
}

template<class flt_t, class acc_t>
void PairDPDIntel::compute(int eflag, int vflag,
        IntelBuffers<flt_t, acc_t> *buffers, const ForceConst<flt_t> &fc) {
    if (eflag || vflag) {
        ev_setup(eflag, vflag);
    } else {
        evflag = vflag_fdotr = 0;
    }
    const int inum = list->inum;
    const int nthreads = comm->nthreads;
    const int host_start = fix->host_start_pair();
    const int offload_end = fix->offload_end_pair();
    const int ago = neighbor->ago;

    if (ago != 0 && fix->separate_buffers() == 0) {
        fix->start_watch(TIME_PACK);

#if defined(_OPENMP)
#pragma omp parallel default(none) shared(eflag,vflag,buffers,fc)
#endif
        {
            int ifrom, ito, tid;
            IP_PRE_omp_range_id_align(ifrom, ito, tid,
                    atom->nlocal + atom->nghost, nthreads, sizeof(ATOM_T));
            buffers->thr_pack(ifrom, ito, ago);
        }
        fix->stop_watch(TIME_PACK);
    }

    if (_onetype) {
        if (evflag || vflag_fdotr) {
            int ovflag = 0;
            if (vflag_fdotr)
                ovflag = 2;
            else if (vflag)
                ovflag = 1;
            if (eflag) {
                if (force->newton_pair) {
                    eval<1, 1, 1, 1>(1, ovflag, buffers, fc, 0, offload_end);
                    eval<1, 1, 1, 1>(0, ovflag, buffers, fc, host_start, inum);
                } else {
                    eval<1, 1, 1, 0>(1, ovflag, buffers, fc, 0, offload_end);
                    eval<1, 1, 1, 0>(0, ovflag, buffers, fc, host_start, inum);
                }
            } else {
                if (force->newton_pair) {
                    eval<1, 1, 0, 1>(1, ovflag, buffers, fc, 0, offload_end);
                    eval<1, 1, 0, 1>(0, ovflag, buffers, fc, host_start, inum);
                } else {
                    eval<1, 1, 0, 0>(1, ovflag, buffers, fc, 0, offload_end);
                    eval<1, 1, 0, 0>(0, ovflag, buffers, fc, host_start, inum);
                }
            }
        } else {
            if (force->newton_pair) {
                eval<1, 0, 0, 1>(1, 0, buffers, fc, 0, offload_end);
                eval<1, 0, 0, 1>(0, 0, buffers, fc, host_start, inum);
            } else {
                eval<1, 0, 0, 0>(1, 0, buffers, fc, 0, offload_end);
                eval<1, 0, 0, 0>(0, 0, buffers, fc, host_start, inum);
            }
        }
    } else {
        if (evflag || vflag_fdotr) {
            int ovflag = 0;
            if (vflag_fdotr)
                ovflag = 2;
            else if (vflag)
                ovflag = 1;
            if (eflag) {
                if (force->newton_pair) {
                    eval<0, 1, 1, 1>(1, ovflag, buffers, fc, 0, offload_end);
                    eval<0, 1, 1, 1>(0, ovflag, buffers, fc, host_start, inum);
                } else {
                    eval<0, 1, 1, 0>(1, ovflag, buffers, fc, 0, offload_end);
                    eval<0, 1, 1, 0>(0, ovflag, buffers, fc, host_start, inum);
                }
            } else {
                if (force->newton_pair) {
                    eval<0, 1, 0, 1>(1, ovflag, buffers, fc, 0, offload_end);
                    eval<0, 1, 0, 1>(0, ovflag, buffers, fc, host_start, inum);
                } else {
                    eval<0, 1, 0, 0>(1, ovflag, buffers, fc, 0, offload_end);
                    eval<0, 1, 0, 0>(0, ovflag, buffers, fc, host_start, inum);
                }
            }
        } else {
            if (force->newton_pair) {
                eval<0, 0, 0, 1>(1, 0, buffers, fc, 0, offload_end);
                eval<0, 0, 0, 1>(0, 0, buffers, fc, host_start, inum);
            } else {
                eval<0, 0, 0, 0>(1, 0, buffers, fc, 0, offload_end);
                eval<0, 0, 0, 0>(0, 0, buffers, fc, host_start, inum);
            }
        }
    }
}

template<int ONETYPE, int EVFLAG, int EFLAG, int NEWTON_PAIR, class flt_t,
        class acc_t>
void PairDPDIntel::eval(const int offload, const int vflag,
        IntelBuffers<flt_t, acc_t> *buffers, const ForceConst<flt_t> &fc,
        const int astart, const int aend) {
    const int inum = aend - astart;
    if (inum == 0)
        return;
    int nlocal, nall, minlocal;
    fix->get_buffern(offload, nlocal, nall, minlocal);

    const int ago = neighbor->ago;
    IP_PRE_pack_separate_buffers(fix, buffers, ago, offload, nlocal, nall);

    ATOM_T* _noalias const x = buffers->get_x(offload);
    ATOM_T* _noalias const v = buffers->get_v(offload);

    const int * _noalias const numneigh = list->numneigh;
    const int * _noalias const cnumneigh = buffers->cnumneigh(list);
    const int * _noalias const firstneigh = buffers->firstneigh(list);
    const flt_t * _noalias const special_lj = fc.special_lj;
    const flt_t dtinvsqrt = 1.0 / sqrt(update->dt);
    const FC_PACKED1_T * _noalias const pk1 = fc.pk1[0];

    const int ntypes = atom->ntypes + 1;
    const int eatom = this->eflag_atom;
    const int rank = comm->me;
    // Determine how much data to transfer
    int x_size, q_size, f_stride, ev_size, separate_flag;
    IP_PRE_get_transfern(ago, NEWTON_PAIR, EVFLAG, EFLAG, vflag, buffers,
            offload, fix, separate_flag, x_size, q_size, ev_size, f_stride);

    int tc;
    FORCE_T* _noalias f_start;
    acc_t * _noalias ev_global;
    IP_PRE_get_buffers(offload, buffers, fix, tc, f_start, ev_global);
    const int nthreads = tc; //openMP or mic
    RanMarsOffload * orandom_thr = this->random_thr;
    int cop = this->_cop;
#ifdef _LMP_INTEL_OFFLOAD
    double *timer_compute = fix->off_watch_pair();
    int *overflow = fix->get_off_overflow_flag();
    if (offload) fix->start_watch(TIME_OFFLOAD_LATENCY);
#pragma offload target(mic:cop) if(offload) \
	    in(special_lj:length(0) alloc_if(0) free_if(0)) \
	    in(pk1:length(0) alloc_if(0) free_if(0)) \
	    in(firstneigh:length(0) alloc_if(0) free_if(0)) \
	    in(cnumneigh:length(0) alloc_if(0) free_if(0)) \
	    in(numneigh:length(0) alloc_if(0) free_if(0)) \
	    in(x,v:length(x_size) alloc_if(0) free_if(0)) \
	    in(overflow:length(0) alloc_if(0) free_if(0)) \
		in(nthreads,inum,nall,ntypes,vflag,eatom,offload,dtinvsqrt) \
		in(astart,nlocal,f_stride,minlocal,separate_flag) \
	    out(f_start:length(f_stride) alloc_if(0) free_if(0)) \
	    out(ev_global:length(ev_size) alloc_if(0) free_if(0)) \
	    out(timer_compute:length(1) alloc_if(0) free_if(0)) \
		in(orandom_thr:length(0) alloc_if(0) free_if(0)) \
		signal(f_start)
#endif
    {
#ifdef __MIC__
        *timer_compute = MIC_Wtime();
#endif

        IP_PRE_repack_for_offload(NEWTON_PAIR, separate_flag, nlocal, nall,
                f_stride, x, 0); //nothing?

        acc_t oevdwl, ov0, ov1, ov2, ov3, ov4, ov5;
        if (EVFLAG) {
            oevdwl = (acc_t) 0;
            if (vflag)
                ov0 = ov1 = ov2 = ov3 = ov4 = ov5 = (acc_t) 0;
        }

        // loop over neighbors of my atoms
#if defined(_OPENMP)
#pragma omp parallel default(none) \
      shared(f_start,f_stride,nlocal,nall,minlocal,orandom_thr) \
      reduction(+:oevdwl,ov0,ov1,ov2,ov3,ov4,ov5)
#endif
        {
            int iifrom, iito, tid;
            IP_PRE_omp_range_id(iifrom, iito, tid, inum, nthreads);
            iifrom += astart;
            iito += astart;

            FORCE_T* _noalias const f = f_start - minlocal + (tid * f_stride);
            memset(f + minlocal, 0, f_stride * sizeof(FORCE_T)); //*2?
            printf(
                    "rank=%d tid=%d/%d, minlocal=%d,f_stride=%d, iifrom=%d, iito=%d\n",
                    rank, tid, nthreads, minlocal, f_stride, iifrom, iito);

            RanMarsOffload & rng = orandom_thr[tid];

            for (int i = iifrom; i < iito; ++i) {
                const int itype = x[i].w;
                const int ptr_off = itype * ntypes;
                const FC_PACKED1_T * _noalias const pk1i = pk1 + ptr_off;

                const int * _noalias const jlist = firstneigh + cnumneigh[i];
                const int jnum = numneigh[i];

                acc_t fxtmp, fytmp, fztmp, fwtmp;
                acc_t sevdwl, sv0, sv1, sv2, sv3, sv4, sv5;

                const flt_t xtmp = x[i].x;
                const flt_t ytmp = x[i].y;
                const flt_t ztmp = x[i].z;

                const flt_t vxtmp = v[i].x;
                const flt_t vytmp = v[i].y;
                const flt_t vztmp = v[i].z;
                fxtmp = fytmp = fztmp = (acc_t) 0;
                if (EVFLAG) {
                    if (EFLAG)
                        fwtmp = sevdwl = (acc_t) 0;
                    if (vflag == 1)
                        sv0 = sv1 = sv2 = sv3 = sv4 = sv5 = (acc_t) 0;
                }

#if defined(LMP_SIMD_COMPILER)
#pragma vector aligned
#pragma simd reduction(+:fxtmp, fytmp, fztmp, fwtmp, sevdwl, \
	                       sv0, sv1, sv2, sv3, sv4, sv5)
#endif
                for (int jj = 0; jj < jnum; jj++) {
                    flt_t evdwl = (flt_t) 0.0;
                    const int sbindex = jlist[jj] >> SBBITS & 3;
                    const int j = jlist[jj] & NEIGHMASK;
                    const flt_t delx = xtmp - x[j].x;
                    const flt_t dely = ytmp - x[j].y;
                    const flt_t delz = ztmp - x[j].z;
                    const int jtype = x[j].w;
                    const flt_t rsq = delx * delx + dely * dely + delz * delz;

                    if (rsq < pk1i[jtype].cutsq) { //must cutoff in DPD
                        flt_t r = sqrt(rsq);
                        if (r < EPSILON)
                            continue; // r can be 0.0 in DPD systems
                        flt_t rinv = 1.0 / r;
                        flt_t factor_dpd = special_lj[sbindex];
                        const flt_t rcut = sqrt(pk1i[jtype].cutsq);
                        const flt_t delvx = vxtmp - v[j].x;
                        const flt_t delvy = vytmp - v[j].y;
                        const flt_t delvz = vztmp - v[j].z;
                        const flt_t dot = delx * delvx + dely * delvy
                                + delz * delvz;
                        const flt_t wd = 1.0 - r / rcut;
                        double randnum = RanMarsOffload_gaussian(&rng);
                        // conservative force = a0 * wd
                        // drag force = -gamma * wd^2 * (delx dot delv) / r
                        // random force = sigma * wd * rnd * dtinvsqrt;

                        flt_t fpair = pk1i[jtype].a0 * wd;
                        fpair -= pk1i[jtype].gamma * wd * wd * dot * rinv;
                        fpair += pk1i[jtype].sigma * wd * randnum * dtinvsqrt;
                        fpair *= factor_dpd * rinv;

                        fxtmp += delx * fpair;
                        fytmp += dely * fpair;
                        fztmp += delz * fpair;
                        if (NEWTON_PAIR || j < nlocal) {
                            f[j].x -= delx * fpair;
                            f[j].y -= dely * fpair;
                            f[j].z -= delz * fpair;
                        }

                        if (EVFLAG) {
                            flt_t ev_pre = (flt_t) 0;
                            if (NEWTON_PAIR || i < nlocal)
                                ev_pre += (flt_t) 0.5;
                            if (NEWTON_PAIR || j < nlocal)
                                ev_pre += (flt_t) 0.5;

                            if (EFLAG) {
                                // unshifted eng of conservative term:
                                // evdwl = -a0[itype][jtype]*r * (1.0-0.5*r/cut[itype][jtype]);
                                // eng shifted to 0.0 at cutoff
                                evdwl = 0.5 * pk1i[jtype].a0 * rcut * wd * wd;
                                evdwl *= factor_dpd;
                                sevdwl += ev_pre * evdwl;
                                if (eatom) {
                                    if (NEWTON_PAIR || i < nlocal)
                                        fwtmp += 0.5 * evdwl;
                                    if (NEWTON_PAIR || j < nlocal)
                                        f[j].w += 0.5 * evdwl;
                                }
                            }

                            IP_PRE_ev_tally_nbor(vflag, ev_pre, fpair, delx,
                                    dely, delz);					//Viral
                        }
                    } // if rsq

                } // for jj
                f[i].x += fxtmp;
                f[i].y += fytmp;
                f[i].z += fztmp;
                IP_PRE_ev_tally_atom(EVFLAG, EFLAG, vflag, f, fwtmp);
            } // for ii

#if defined(_OPENMP)
#pragma omp barrier
#endif
            IP_PRE_fdotr_acc_force(NEWTON_PAIR, EVFLAG, EFLAG, vflag, eatom,
                    nall, nlocal, minlocal, nthreads, f_start, f_stride, x,
                    offload);
        } // end omp
        if (EVFLAG) {
            if (EFLAG) {
                ev_global[0] = oevdwl;
                ev_global[1] = (acc_t) 0.0;
            }
            if (vflag) {
                ev_global[2] = ov0;
                ev_global[3] = ov1;
                ev_global[4] = ov2;
                ev_global[5] = ov3;
                ev_global[6] = ov4;
                ev_global[7] = ov5;
            }
        }
#if defined(__MIC__) && defined(_LMP_INTEL_OFFLOAD)
        *timer_compute = MIC_Wtime() - *timer_compute;
#endif
    } // end offload

    if (offload)
        fix->stop_watch(TIME_OFFLOAD_LATENCY);
    else
        fix->stop_watch(TIME_HOST_PAIR);

    if (EVFLAG)
        fix->add_result_array(f_start, ev_global, offload, eatom, 0, vflag);
    else
        fix->add_result_array(f_start, 0, offload);
}

/* ---------------------------------------------------------------------- */

void PairDPDIntel::init_style() {
    PairDPD::init_style();
    neighbor->requests[neighbor->nrequest - 1]->intel = 1;

    int ifix = modify->find_fix("package_intel");
    if (ifix < 0)
        error->all(FLERR,
                "The 'package intel' command is required for /intel styles");
    fix = static_cast<FixIntel *>(modify->fix[ifix]);

    fix->pair_init_check();
#ifdef _LMP_INTEL_OFFLOAD
    if (force->newton_pair) fix->set_offload_noghost(1);
    _cop = fix->coprocessor_number();
#endif

    if (fix->precision() == FixIntel::PREC_MODE_MIXED)
        pack_force_const(force_const_single, fix->get_mixed_buffers());
    else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE)
        pack_force_const(force_const_double, fix->get_double_buffers());
    else
        pack_force_const(force_const_single, fix->get_single_buffers());
    const int nthreads = comm->nthreads;
    const int seedme = seed + comm->me;
    const int nprocs = comm->nprocs;
    printf("set MPI task threads=%d\n", nthreads);
    RanMarsOffload * orandom_thr;
    int othreads = this->off_threads;
    if (random_thr) { //run again, try free it
        printf("rank =%d release RanMarsOffload[%d] \n", comm->me, off_threads);
        orandom_thr = this->random_thr;
#ifdef _LMP_INTEL_OFFLOAD
#pragma offload_transfer target(mic:_cop) if(othreads>0) \
	    nocopy(orandom_thr:length(othreads) alloc_if(0) free_if(1))
#endif
        memory->destroy(random_thr);
    }
    memory->create(random_thr, nthreads, "RanMarsOffload");	//new cpu side
    if (random_thr) {
        // generate a random number generator instance for
        // all threads != 0. make sure we use unique seeds.
        for (int i = 0; i < nthreads; ++i) {
            RanMarsOffload_init(&random_thr[i], seedme + nprocs * i);
        }
    }

#ifdef _LMP_INTEL_OFFLOAD
    if (_cop < 0) {
        error->warning(FLERR, "_cop<0 in Offload mode.");
        return;
    }
//if(comm->me==0) {
    printf("rank =%d set off_threads= %d offload balance=%g random_thr @ %p\n",
            comm->me, off_threads,fix->offload_balance(),random_thr);
//}
    if (random_thr) {
        //alloc on mic with off_threads but no comm->nthreads
        orandom_thr = this->random_thr;
#pragma offload target(mic:_cop) if(othreads>0) \
		in(seedme,nprocs,othreads) \
		nocopy(orandom_thr:length(othreads) alloc_if(1) free_if(0))
        {
            for (int i = 0; i < othreads; ++i) {
                RanMarsOffload_init(&orandom_thr[i], 2*seedme + nprocs * i);
            }
        }
    }
#endif

}

/* ---------------------------------------------------------------------- */

template<class flt_t, class acc_t>
void PairDPDIntel::pack_force_const(ForceConst<flt_t> &fc,
        IntelBuffers<flt_t, acc_t> *buffers) {
    _onetype = 0;
    if (atom->ntypes == 1 && !atom->molecular)
        _onetype = 1;

    int tp1 = atom->ntypes + 1;
    fc.set_ntypes(tp1, memory, _cop);
    buffers->set_ntypes(tp1);
    flt_t **cutneighsq = buffers->get_cutneighsq();

// Repeat cutsq calculation because done after call to init_style
    double rcut, cutneigh;
    for (int i = 1; i <= atom->ntypes; i++) {
        for (int j = i; j <= atom->ntypes; j++) {
            if (setflag[i][j] != 0
                    || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
                rcut = init_one(i, j);
                cutneigh = rcut + neighbor->skin;
                cutsq[i][j] = cutsq[j][i] = rcut * rcut;
                cutneighsq[i][j] = cutneighsq[j][i] = cutneigh * cutneigh;
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        fc.special_lj[i] = force->special_lj[i];
        fc.special_lj[0] = 1.0;
    }

    for (int i = 0; i < tp1; i++) {
        for (int j = 0; j < tp1; j++) {
            fc.pk1[i][j].a0 = a0[i][j];
            fc.pk1[i][j].gamma = gamma[i][j];
            fc.pk1[i][j].sigma = sigma[i][j];
            fc.pk1[i][j].cutsq = cutsq[i][j];
        }
    }
#ifdef _LMP_INTEL_OFFLOAD
    if (_cop < 0) return;
    off_threads=buffers->get_off_threads();
    flt_t * special_lj = fc.special_lj;
    FC_PACKED1_T *opk1 = fc.pk1[0];
    flt_t * ocutneighsq = cutneighsq[0];
    int tp1sq = tp1 * tp1;
    int cop=this->_cop;
    if (opk1 != NULL) {
#pragma offload_transfer target(mic:cop) \
      in(special_lj: length(4) alloc_if(0) free_if(0)) \
      in(opk1: length(tp1sq) alloc_if(0) free_if(0)) \
      in(ocutneighsq: length(tp1sq))
    }
#endif
}

/* ---------------------------------------------------------------------- */

template<class flt_t>
void PairDPDIntel::ForceConst<flt_t>::set_ntypes(const int ntypes,
        Memory *memory, const int cop) {
    if (ntypes != _ntypes) {
        if (_ntypes > 0) {
            fc_packed1 *opk1 = pk1[0];

#ifdef _LMP_INTEL_OFFLOAD
            flt_t * ospecial_lj = special_lj;
            if (ospecial_lj != NULL && _cop >= 0) {
#pragma offload_transfer target(mic:_cop) \
          nocopy(ospecial_lj, opk1: alloc_if(0) free_if(1))
            }
#endif

            _memory->destroy(opk1);
        }
        if (ntypes > 0) {
            _cop = cop;
            memory->create(pk1, ntypes, ntypes, "fc.pk1");

#ifdef _LMP_INTEL_OFFLOAD
            flt_t * ospecial_lj = special_lj;
            fc_packed1 *opk1 = pk1[0];
            int tp1sq = ntypes * ntypes;
            if (ospecial_lj != NULL && opk1 != NULL && cop >= 0) {
#pragma offload_transfer target(mic:cop) \
          nocopy(ospecial_lj: length(4) alloc_if(1) free_if(0)) \
          nocopy(opk1: length(tp1sq) alloc_if(1) free_if(0))
            }
#endif
        }
    }
    _ntypes = ntypes;
    _memory = memory;
}
