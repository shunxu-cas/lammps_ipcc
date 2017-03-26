/* -*- c++ -*- ----------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 Contributing authors: Shun Xu (Computer Network Information Center, CAS)
                       W. Michael Brown (Intel)
 ------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(dpd/intel,PairDPDIntel)

#else

#ifndef LMP_PAIR_DPD_INTEL_H
#define LMP_PAIR_DPD_INTEL_H

#include "pair_dpd.h"
#include "fix_intel.h"

namespace LAMMPS_NS {

typedef struct {
    int save;
    int seed;
    double second;
    double u[98]; //97 + 1
    int i97, j97;
    double c, cd, cm;
} RanMarsOffload;

class PairDPDIntel: public PairDPD {

public:
    PairDPDIntel(class LAMMPS *);
    virtual ~PairDPDIntel();
    virtual void compute(int, int);
    void init_style();

protected:
    RanMarsOffload *random_thr;
    int both_nthreads;
    void free_rand_thr();

private:
    FixIntel *fix;
    int _cop, _onetype;

    template<class flt_t> class ForceConst;
    template<class flt_t, class acc_t>
    void compute(int eflag, int vflag, IntelBuffers<flt_t, acc_t> *buffers,
            const ForceConst<flt_t> &fc);
    template<int ONETYPE, int EVFLAG, int EFLAG, int NEWTON_PAIR, class flt_t,
            class acc_t>
    void eval(const int offload, const int vflag,
            IntelBuffers<flt_t, acc_t> * buffers, const ForceConst<flt_t> &fc,
            const int astart, const int aend);

    template<class flt_t, class acc_t>
    void pack_force_const(ForceConst<flt_t> &fc,
            IntelBuffers<flt_t, acc_t> *buffers);

    // ----------------------------------------------------------------------

    template<class flt_t>
    class ForceConst {
    public:
        typedef struct {
            flt_t cutsq, a0, gamma, sigma;
        } fc_packed1;
        typedef struct {
            flt_t lj3, lj4;
        } fc_packed2;

        /* special_lj[4] is a static type, retain offloading values by default ! */
        //_alignvar(flt_t special_lj[4],64);
        flt_t * special_lj;
        fc_packed1 **pk1;
        //fc_packed2 **lj34;

        ForceConst() :
                _ntypes(0) {
            special_lj = NULL;
            pk1=NULL;
        }
        ~ForceConst() {
            free_offload(_cop);
        }
        void free_offload(const int cop);
        void set_ntypes(const int ntypes, Memory *memory, const int cop);

    private:
        int _ntypes, _cop;
        Memory *_memory;
    };
    ForceConst<float> force_const_single;
    ForceConst<double> force_const_double;
};

}

#endif
#endif

/* ERROR/WARNING messages:

 E: The 'package intel' command is required for /intel styles

 Self-explanatory.

 */
