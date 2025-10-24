/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2020, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "popsift/sift_config.h"

#include "common/assist.h"
#include "common/debug_macros.h"
#include "common/vec_macros.h"
#include "s_desc_vlfeat.h"
#include "s_gradiant.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cstdio>

using namespace popsift;

// static inline
// void ext_desc_vlfeat_sub( const float         ang,
//                           const Extremum*     ext,
//                           float* __restrict__ features,
//                           Plane2D_float&      layer_tex,
//                           const int           width,
//                           const int           height )
// {
//     const float x     = ext->xpos;
//     const float y     = ext->ypos;
//     const int   level = ext->lpos; // old_level;
//     const float sig   = ext->sigma;
//     const float SBP   = fabsf(DESC_MAGNIFY * sig);

//     if( SBP == 0 ) {
//         return;
//     }

//     float cos_t;
//     float sin_t;
// #ifndef __APPLE__
//     sincosf( ang, &sin_t, &cos_t );
// #else
//     __sincosf( ang, &sin_t, &cos_t );
// #endif

//     const float csbp  = cos_t * SBP;
//     const float ssbp  = sin_t * SBP;
//     const float crsbp = cos_t / SBP;
//     const float srsbp = sin_t / SBP;

//     // We have 4x4*16 bins.
//     // There centers have the offsets -1.5, -0.5, 0.5, 1.5 from the
//     // keypoint. The points that support them stretch from -2 to 2
//     const float2 maxdist = make_float2( -2.0f, -2.0f );

//     // We rotate the corner of the maximum range by the keypoint orientation.
//     // const float ptx = csbp * maxdist - ssbp * maxdist;
//     // const float pty = csbp * maxdist + ssbp * maxdist;
//     const float ptx = fabsf( ::fmaf( csbp, maxdist.x, -ssbp * maxdist.y ) );
//     const float pty = fabsf( ::fmaf( csbp, maxdist.y,  ssbp * maxdist.x ) );

//     const float bsz = 2.0f * ( fabsf(csbp) + fabsf(ssbp) );

//     const int   xmin = std::max(1,          (int)floorf(x - ptx - bsz));
//     const int   ymin = std::max(1,          (int)floorf(y - pty - bsz));
//     const int   xmax = std::min(width - 2,  (int)floorf(x + ptx + bsz));
//     const int   ymax = std::min(height - 2, (int)floorf(y + pty + bsz));

//     float dpt[128] = { 0 };

//     for( int pix_y = ymin; pix_y <= ymax; pix_y += 1 )
//     {
//         for( int pix_x = xmin; pix_x <= xmax; pix_x += 1 )
//         {
//             float mod;
//             float th;

//             // CG: the gradient's X input may be wrong in the CUDA version, check that
//             get_gradiant32( mod, th, pix_x, pix_y, layer_tex, level );

//             mod /= 2.0f; // Our mod is double that of vlfeat. Huh.

//             th -= ang;
//             if( th < -3 || th > 3 )
//             {
//                 const int x=pix_x;
//                 const int y=pix_y;
//                 const float pix_x_pls_1 = layer_tex.get( level, y  , x+1 );
//                 const float pix_x_min_1 = layer_tex.get( level, y  , x-1 );
//                 const float pix_y_pls_1 = layer_tex.get( level, y+1, x   );
//                 const float pix_y_min_1 = layer_tex.get( level, y-1, x   );
//                 const float dx          = pix_x_pls_1 - pix_x_min_1;
//                 const float dy          = pix_y_pls_1 - pix_y_min_1;
// #if 0
//                 std::cerr << __LINE__ << ":"
//                         << "    computation after get_gradient32 from (x=" << pix_x << ", y=" << pix_y << ") yields mod=" << mod << ", th=" << th
//                         << std::endl;
// #endif
//             }

//             while( th > M_PI2 ) {
//                 th -= M_PI2;
//             }
//             while( th < 0.0f  ) {
//                 th += M_PI2;
//             }

//             // d : distance from keypoint
//             const float2 d = make_float2( pix_x - x, pix_y - y );

//             // n : normalized distance from keypoint
//             const float2 n = make_float2( ::fmaf( crsbp, d.x,  srsbp * d.y ),
//                                         ::fmaf( crsbp, d.y, -srsbp * d.x ) ); 

//             const float  ww = expf( -scalbnf(n.x*n.x + n.y*n.y, -3));

//             const float nt = 8.0f * th / M_PI2;

//             // neighbouring tile on the lower side: -2, -1, 0 or 1
//             // (must use floorf because casting rounds towards zero
//             const int3 t0 = make_int3( (int)floorf(n.x - 0.5f),
//                                     (int)floorf(n.y - 0.5f),
//                                     (int)nt );
//             const float wgt_x = - ( n.x - ( t0.x + 0.5f ) );
//             const float wgt_y = - ( n.y - ( t0.y + 0.5f ) );
//             const float wgt_t = - ( nt  - t0.z );

//             for( int tx=0; tx<2; tx++ )
//             {
//                 for( int ty=0; ty<2; ty++ )
//                 {
//                     for( int tt=0; tt<2; tt++ )
//                     {
//                         if( ( t0.y + ty >= -2 ) &&
//                             ( t0.y + ty <   2 ) &&
//                             ( t0.x + tx >= -2 ) &&
//                             ( t0.x + tx <   2 ) )
//                         {
//                             float i_wgt_x = ( tx == 0 ) ? 1.0f + wgt_x : wgt_x;
//                             float i_wgt_y = ( ty == 0 ) ? 1.0f + wgt_y : wgt_y;
//                             float i_wgt_t = ( tt == 0 ) ? 1.0f + wgt_t : wgt_t;

//                             i_wgt_x = fabsf( i_wgt_x );
//                             i_wgt_y = fabsf( i_wgt_y );
//                             i_wgt_t = fabsf( i_wgt_t );

//                             const float val = ww
//                                             * mod
//                                             * i_wgt_x
//                                             * i_wgt_y
//                                             * i_wgt_t;

//                             const int offset =  80
//                                             + ( t0.y + ty ) * 32
//                                             + ( t0.x + tx ) * 8
//                                             + ( t0.z + tt ) % 8;

//                             dpt[offset] += val;
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     for( int i=0; i<128; i++ )
//     {
//         features[i] = dpt[i];
//     }
// }

// void ext_desc_vlfeat( int octave, Plane2D_float& layer_tex, int w, int h)
// {
//     const int num_orientations = dct.ori_ct[octave];

// #if 1
//     std::cerr << "Exclusive prefix sum of orientations (dct.ori_ps): ";
//     for( int i=0; i<MAX_OCTAVES; i++ ) std::cerr << dct.ori_ps[i] << " ";
//     std::cerr << std::endl;
// #endif
//     for( int ori_idx=0; ori_idx<num_orientations; ori_idx++ )
//     {
//         std::cerr << "Extract orientation " << ori_idx << " from octave " << octave << std::endl;
//         std::cerr << "    Start offset of orientations for octave " << octave << " in global orientation array: " << dct.ori_ps[octave] << std::endl;
//         const int   o_offset =  dct.ori_ps[octave] + ori_idx;
//         std::cerr << "    Octave's global orientation offset: " << o_offset << std::endl;
//         Descriptor* desc     = &dbuf.desc           [o_offset];
//         const int   ext_idx  =  dbuf.feat_to_ext_map[o_offset];
//         std::cerr << "    Feature index: " << ext_idx << std::endl;
//         Extremum*   ext      = &dbuf.extrema[ext_idx];

//         const int   ext_base =  ext->idx_ori;
//         const int   ori_num  =  o_offset - ext_base;
//         const float ang      =  ext->orientation[ori_num];

//         std::cerr << "    Calling vlfeat_sub with "
//                   << "pos=(" << ext->xpos << "," << ext->ypos << "," << ext->lpos << ") "
//                   << "sigma=" << std::fixed << std::setprecision(3) << ext->sigma << " "
//                   << "octave=" << ext->octave << " "
//                   << "ang=" << ang << " (" << std::fixed << std::setprecision(3) << (ang / M_PI2 * 360.0f) << " degrees)" << std::endl;

//         ext_desc_vlfeat_sub( ang,
//                              ext,
//                              desc->features,
//                              layer_tex,
//                              w,
//                              h );
//         std::cerr << "    Done" << std::endl;
//     }
// }

// namespace popsift
// {

// bool start_ext_desc_vlfeat( const int octave, Octave& oct_obj )
// {
// #if 1
//     std::cerr << __FUNCTION__ << " computing descriptors for octave " << octave << std::endl;
//     std::cerr << "    number of orientations: " << dct.ori_ct[octave] << std::endl;
// #endif
//     if( dct.ori_ct[octave] == 0 ) return false;

//     ext_desc_vlfeat
//         ( octave,
//           oct_obj.getData( ),
//           oct_obj.getWidth(),
//           oct_obj.getHeight() );

//     return true;
// }

// }; // namespace popsift

namespace popsift
{

// SYCL kernel for descriptor extraction
static void ext_desc_vlfeat_sycl(
    sycl::queue& q,
    const int octave,
    const float* d_layer_data,
    const int layer_pitch,
    const int num_levels,
    const int width,
    const int height,
    const Extremum* d_extrema,
    const int* d_feat_to_ext_map,
    const int num_orientations,
    const int orientation_offset,
    Descriptor* d_descriptors)
{
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_orientations), [=](sycl::id<1> idx) {
            const int ori_idx = idx[0];
            const int o_offset = orientation_offset + ori_idx;
            const int ext_idx = d_feat_to_ext_map[o_offset];
            const Extremum ext = d_extrema[ext_idx];
            
            const int ext_base = ext.idx_ori;
            const int ori_num = o_offset - ext_base;
            const float ang = ext.orientation[ori_num];
            
            const float x = ext.xpos;
            const float y = ext.ypos;
            const int level = ext.lpos;
            const float sig = ext.sigma;
            const float SBP = sycl::fabs(DESC_MAGNIFY * sig);
            
            if(SBP == 0.0f) return;
            
            float cos_t = sycl::cos(ang);
            float sin_t = sycl::sin(ang);
            
            const float csbp = cos_t * SBP;
            const float ssbp = sin_t * SBP;
            const float crsbp = cos_t / SBP;
            const float srsbp = sin_t / SBP;
            
            const float maxdist_x = -2.0f;
            const float maxdist_y = -2.0f;
            
            const float ptx = sycl::fabs(csbp * maxdist_x - ssbp * maxdist_y);
            const float pty = sycl::fabs(csbp * maxdist_y + ssbp * maxdist_x);
            
            const float bsz = 2.0f * (sycl::fabs(csbp) + sycl::fabs(ssbp));
            
            const int xmin = sycl::max(1, (int)sycl::floor(x - ptx - bsz));
            const int ymin = sycl::max(1, (int)sycl::floor(y - pty - bsz));
            const int xmax = sycl::min(width - 2, (int)sycl::floor(x + ptx + bsz));
            const int ymax = sycl::min(height - 2, (int)sycl::floor(y + pty + bsz));
            
            float dpt[128];
            for(int i = 0; i < 128; i++) dpt[i] = 0.0f;
            
            // Bounds check for level and ensure we can access neighbors
            if(level < 0 || level >= num_levels) return;
            
            // Sample gradients in the patch
            for(int pix_y = ymin; pix_y <= ymax; pix_y++) {
                for(int pix_x = xmin; pix_x <= xmax; pix_x++) {
                    // CRITICAL: We need to access pix_x±1 and pix_y±1 for gradients
                    // So ensure these are within bounds
                    if(pix_x <= 0 || pix_x >= width - 1 || 
                       pix_y <= 0 || pix_y >= height - 1) {
                        continue;
                    }
                    
                    // Now we know pix_x-1, pix_x+1, pix_y-1, pix_y+1 are all valid
                    const int offset = level * (height * layer_pitch) + pix_y * layer_pitch + pix_x;
                    
                    const float pix_x_pls_1 = d_layer_data[offset + 1];
                    const float pix_x_min_1 = d_layer_data[offset - 1];
                    const float pix_y_pls_1 = d_layer_data[offset + layer_pitch];
                    const float pix_y_min_1 = d_layer_data[offset - layer_pitch];
                    
                    const float dx = pix_x_pls_1 - pix_x_min_1;
                    const float dy = pix_y_pls_1 - pix_y_min_1;
                    
                    float mod = sycl::sqrt(dx * dx + dy * dy);
                    float th = sycl::atan2(dy, dx);
                    
                    mod /= 2.0f;
                    
                    th -= ang;
                    while(th > M_PI2) th -= M_PI2;
                    while(th < 0.0f) th += M_PI2;
                    
                    const float d_x = pix_x - x;
                    const float d_y = pix_y - y;
                    
                    const float n_x = crsbp * d_x + srsbp * d_y;
                    const float n_y = crsbp * d_y - srsbp * d_x;
                    
                    const float ww = sycl::exp(-(n_x * n_x + n_y * n_y) / 8.0f);
                    
                    const float nt = 8.0f * th / M_PI2;
                    
                    const int t0_x = (int)sycl::floor(n_x - 0.5f);
                    const int t0_y = (int)sycl::floor(n_y - 0.5f);
                    const int t0_z = (int)nt;
                    
                    const float wgt_x = -(n_x - (t0_x + 0.5f));
                    const float wgt_y = -(n_y - (t0_y + 0.5f));
                    const float wgt_t = -(nt - t0_z);
                    
                    for(int tx = 0; tx < 2; tx++) {
                        for(int ty = 0; ty < 2; ty++) {
                            for(int tt = 0; tt < 2; tt++) {
                                if((t0_y + ty >= -2) && (t0_y + ty < 2) &&
                                   (t0_x + tx >= -2) && (t0_x + tx < 2)) {
                                    
                                    float i_wgt_x = (tx == 0) ? 1.0f + wgt_x : wgt_x;
                                    float i_wgt_y = (ty == 0) ? 1.0f + wgt_y : wgt_y;
                                    float i_wgt_t = (tt == 0) ? 1.0f + wgt_t : wgt_t;
                                    
                                    i_wgt_x = sycl::fabs(i_wgt_x);
                                    i_wgt_y = sycl::fabs(i_wgt_y);
                                    i_wgt_t = sycl::fabs(i_wgt_t);
                                    
                                    const float val = ww * mod * i_wgt_x * i_wgt_y * i_wgt_t;
                                    
                                    const int offset_desc = 80 + (t0_y + ty) * 32 + (t0_x + tx) * 8 + (t0_z + tt) % 8;
                                    
                                    dpt[offset_desc] += val;
                                }
                            }
                        }
                    }
                }
            }
            
            // Write descriptor
            Descriptor* desc = &d_descriptors[o_offset];
            for(int i = 0; i < 128; i++) {
                desc->features[i] = dpt[i];
            }
        });
    }).wait();
}
bool start_ext_desc_vlfeat( const int octave, Octave& oct_obj )
{
    std::cerr << __FUNCTION__ << " computing descriptors for octave " << octave << std::endl;
    std::cerr << "    number of orientations: " << dct.ori_ct[octave] << std::endl;
    
    if( dct.ori_ct[octave] == 0 ) return false;

    const int num_orientations = dct.ori_ct[octave];
    
    sycl::queue& q = oct_obj.getQueue();
    
    std::cerr << "    Octave info: width=" << oct_obj.getWidth() 
              << " height=" << oct_obj.getHeight()
              << " pitch=" << oct_obj.getData().getCols()
              << " levels=" << oct_obj.getLevels() << std::endl;
    
    // Check if dbuf.desc is a host or device pointer
    std::cerr << "    dbuf.desc pointer: " << (void*)dbuf.desc << std::endl;
    std::cerr << "    dct.ori_total: " << dct.ori_total << std::endl;
    
    // Test: Try to write to dbuf.desc from CPU - if it crashes, it's a device pointer
    try {
        dbuf.desc[0].features[0] = 123.456f;
        std::cerr << "    dbuf.desc is ACCESSIBLE FROM CPU (host memory)" << std::endl;
        std::cerr << "    Value written: " << dbuf.desc[0].features[0] << std::endl;
    } catch(...) {
        std::cerr << "    dbuf.desc is NOT accessible from CPU (device memory)" << std::endl;
    }
    
    // Allocate DEVICE memory for descriptors
    std::cerr << "    Allocating device memory for " << dct.ori_total << " descriptors" << std::endl;
    Descriptor* d_desc = sycl::malloc_device<Descriptor>(dct.ori_total, q);
    std::cerr << "    d_desc device pointer: " << (void*)d_desc << std::endl;
    
    // Allocate device memory and copy data
    Extremum* d_extrema = sycl::malloc_device<Extremum>(dbuf.extrema.size(), q);
    int* d_feat_to_ext_map = sycl::malloc_device<int>(dbuf.feat_to_ext_map.size(), q);
    
    std::cerr << "    Copying extrema and mappings to device" << std::endl;
    q.memcpy(d_extrema, dbuf.extrema.data(), dbuf.extrema.size() * sizeof(Extremum)).wait();
    q.memcpy(d_feat_to_ext_map, dbuf.feat_to_ext_map.data(), dbuf.feat_to_ext_map.size() * sizeof(int)).wait();
    
    std::cerr << "    Launching SYCL descriptor kernel" << std::endl;
    
    ext_desc_vlfeat_sycl(
        q,
        octave,
        oct_obj.getData().getDevPtr(),
        oct_obj.getData().getCols(),
        oct_obj.getLevels(),
        oct_obj.getWidth(),
        oct_obj.getHeight(),
        d_extrema,
        d_feat_to_ext_map,
        1,  // ONLY PROCESS FIRST ORIENTATION
        dct.ori_ps[octave],
        d_desc  // Use our allocated device memory instead of dbuf.desc
    );
    
    std::cerr << "    SYCL kernel completed successfully!" << std::endl;
    
    // Copy results back to host (dbuf.desc)
    std::cerr << "    Copying results back to host" << std::endl;
    q.memcpy(dbuf.desc, d_desc, num_orientations * sizeof(Descriptor)).wait();
    
    std::cerr << "    First descriptor value: " << dbuf.desc[dct.ori_ps[octave]].features[0] << std::endl;
    
    // Clean up
    sycl::free(d_extrema, q);
    sycl::free(d_feat_to_ext_map, q);
    sycl::free(d_desc, q);

    return true;
}


}; // namespace popsift