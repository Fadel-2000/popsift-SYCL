/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/clamp.h"
#include "common/debug_macros.h"
#include "common/grid.h"
#include "s_solve.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <sys/stat.h>

#include <cstdio>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iterator>
#include <map> 

namespace popsift{

static
inline void extremum_cmp( float val, float f, uint32_t& gt, uint32_t& lt, uint32_t mask )
{
    gt |= ( ( val > f ) ? mask : 0 );
    lt |= ( ( val < f ) ? mask : 0 );
}

#define TX(dx,dy,dz) obj.get( z+dz, y+dy, x+dx )

static
inline bool is_extremum( Plane2D_float& obj,
                         int x, int y, int z )
{
    uint32_t gt = 0;
    uint32_t lt = 0;

    const float val0 = TX( 0, 1, 1 );
    const float val2 = TX( 2, 1, 1 );
    const float val  = TX( 1, 1, 1 );

    // bit indeces for neighbours:
    //     7 0 1    0x80 0x01 0x02
    //     6   2 -> 0x40      0x04
    //     5 4 3    0x20 0x10 0x08
    // upper layer << 24 ; own layer << 16 ; lower layer << 8
    // 1st group: left and right neigbhour
    extremum_cmp( val, val0, gt, lt, 0x00400000 ); // ( 0x01<<6 ) << 16
    extremum_cmp( val, val2, gt, lt, 0x00040000 ); // ( 0x01<<2 ) << 16

    if( ( gt != 0x00440000 ) && ( lt != 0x00440000 ) ) return false;

    // 2nd group: requires a total of 8 128-byte reads
    extremum_cmp( val, TX(0,0,1), gt, lt, 0x00800000 ); // ( 0x01<<7 ) << 16
    extremum_cmp( val, TX(0,2,1), gt, lt, 0x00200000 ); // ( 0x01<<5 ) << 16
    extremum_cmp( val, TX(0,0,0), gt, lt, 0x80000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, TX(0,2,0), gt, lt, 0x40000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, TX(0,1,0), gt, lt, 0x20000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, TX(0,0,2), gt, lt, 0x00008000 ); // ( 0x01<<6 ) <<  8
    extremum_cmp( val, TX(0,1,2), gt, lt, 0x00004000 ); // ( 0x01<<6 ) <<  8
    extremum_cmp( val, TX(0,2,2), gt, lt, 0x00002000 ); // ( 0x01<<6 ) <<  8

    if( ( gt != 0xe0e4e000 ) && ( lt != 0xe0e4e000 ) ) return false;

    // 3rd group: remaining 2 cache misses in own layer
    extremum_cmp( val, TX(1,0,1), gt, lt, 0x00010000 ); // ( 0x01<<0 ) << 16
    extremum_cmp( val, TX(2,0,1), gt, lt, 0x00020000 ); // ( 0x01<<1 ) << 16
    extremum_cmp( val, TX(1,2,1), gt, lt, 0x00100000 ); // ( 0x01<<4 ) << 16
    extremum_cmp( val, TX(2,2,1), gt, lt, 0x00080000 ); // ( 0x01<<3 ) << 16

    if( ( gt != 0xe0ffe000 ) && ( lt != 0xe0ffe000 ) ) return false;

    // 4th group: 3 cache misses higher layer
    extremum_cmp( val, TX(1,0,0), gt, lt, 0x01000000 ); // ( 0x01<<0 ) << 24
    extremum_cmp( val, TX(2,0,0), gt, lt, 0x02000000 ); // ( 0x01<<1 ) << 24
    extremum_cmp( val, TX(1,1,0), gt, lt, 0x00000004 ); // ( 0x01<<2 )
    extremum_cmp( val, TX(2,1,0), gt, lt, 0x04000000 ); // ( 0x01<<2 ) << 24
    extremum_cmp( val, TX(1,2,0), gt, lt, 0x10000000 ); // ( 0x01<<4 ) << 24
    extremum_cmp( val, TX(2,2,0), gt, lt, 0x08000000 ); // ( 0x01<<3 ) << 24

    if( ( gt != 0xffffe004 ) && ( lt != 0xffffe004 ) ) return false;

    // 5th group: 3 cache misss lower layer
    extremum_cmp( val, TX(1,0,2), gt, lt, 0x00000100 ); // ( 0x01<<0 ) <<  8
    extremum_cmp( val, TX(2,0,2), gt, lt, 0x00000200 ); // ( 0x01<<1 ) <<  8
    extremum_cmp( val, TX(1,1,2), gt, lt, 0x00000001 ); // ( 0x01<<0 )
    extremum_cmp( val, TX(2,1,2), gt, lt, 0x00000400 ); // ( 0x01<<2 ) <<  8
    extremum_cmp( val, TX(1,2,2), gt, lt, 0x00001000 ); // ( 0x01<<4 ) <<  8
    extremum_cmp( val, TX(2,2,2), gt, lt, 0x00000800 ); // ( 0x01<<3 ) <<  8

    if( ( gt != 0xffffff05 ) && ( lt != 0xffffff05 ) ) return false;

    return true;
}

template<int sift_mode>
class ModeFunctions
{
public:
    /* refine
     * returns 0 : continue looping
     *         1 : break loop and succeed
     */
    inline 
    int refine( float3& d, int3& n, int width, int height, int maxlevel, bool last_it );
};

template<>
class ModeFunctions<Config::RefineInLevel>
{
public:
    inline 
    int refine( float3& d, int3& n, int width, int height, int maxlevel, bool last_it ) const
    {
        if( last_it ) return 0;

        int2 t;

        t.x = ((d.x >=  0.6f && n.x < width-2) ?  1 : 0 )
            + ((d.x <= -0.6f && n.x > 1)       ? -1 : 0 );

        t.y = ((d.y >=  0.6f && n.y < height-2)  ?  1 : 0 )
            + ((d.y <= -0.6f && n.y > 1)         ? -1 : 0 );

        if( t.x == 0 && t.y == 0 ) {
            // no more changes
            return 1;
        }

        n.x += t.x;
        n.y += t.y;
        // n.z += t.z; - VLFeat is not changing levels !!!

        return 0;
    }
};

template<>
class ModeFunctions<Config::RefineInOctave>
{
public:
    inline 
    int refine( float3& d, int3& n, int width, int height, int maxlevel, bool last_it ) const
    {
        if( last_it ) return 0;

        int3 t;

        t.x = ((d.x >=  0.6f && n.x < width-2) ?  1 : 0 )
            + ((d.x <= -0.6f && n.x > 1)       ? -1 : 0 );

        t.y = ((d.y >=  0.6f && n.y < height-2)  ?  1 : 0 )
            + ((d.y <= -0.6f && n.y > 1)         ? -1 : 0 );

        t.z = ((d.z >=  0.6f && n.z < maxlevel-1)  ?  1 : 0 )
            + ((d.z <= -0.6f && n.z > 1)           ? -1 : 0 );

        if( t.x == 0 && t.y == 0 && t.z == 0 ) {
            // no more changes
            return 1;
        }

        n.x += t.x;
        n.y += t.y;
        n.z += t.z;

        return 0;
    }
};

inline static
bool first_contrast_ok( const float val )
{
    return ( fabsf( val ) >= 1.6f * h_consts.threshold );
}

/** verify() checks whether a refine position is outside the image boundaries or
 *  outside the DoG boundaries.
 *  returns true  : values after refine make sense
 *          false : they do not
 */
inline static
bool verify( float xn, float yn, float sn, int width, int height, int maxlevel )
{
    // reject if outside of image bounds or far outside DoG bounds
    return ( ( xn < 0.0f ||
               xn > width - 1.0f ||
               yn < 0.0f ||
               yn > height - 1.0f ||
               sn < -0.0f ||
               sn > maxlevel ) ? false
                               : true );
}

template<int sift_mode>
static inline
bool find_extrema_in_dog_sub( const int3&      g,
                              Plane2D_float&   dog,
                              int              this_octave,
                              int              width,
                              int              height,
                              uint32_t         maxlevel,
                              float            w_grid_divider,
                              float            h_grid_divider,
                              int              grid_width,
                              InitialExtremum& ec)
{
    const bool no_extrema_reporting = true;

    ec.xpos    = 0.0f;
    ec.ypos    = 0.0f;
    ec.lpos    = 0;
    ec.sigma   = 0.0f;

    const int x     = g.x + 1;
    const int y     = g.y + 1;
    const int level = g.z + 1;

    const float val = dog.get( level, y, x );

    ModeFunctions<sift_mode> f;
    if( ! first_contrast_ok( val ) ) return false;

    if( ! is_extremum( dog, x-1, y-1, level-1 ) ) {
        return false;
    } else {
        POP_INFO2( no_extrema_reporting, "Found an extremum in octave " << this_octave << " at (" << x << ", " << y << ", " << level << ")" );
    }

    // REMOVE THE DUPLICATED CODE FROM HERE
    // The CPU version should use the original refinement code that was here before
    
    float v = val;
    float3 D;
    float3 DD;
    float3 DX;
    float3 d;
    int3 n;
    n.x = x;
    n.y = y;
    n.z = level;

    for( int iter=0; iter<5; iter++ )
    {
        D.x = scalbnf( dog.get( n.z, n.y, n.x+1 ) - dog.get( n.z, n.y, n.x-1 ), -1 );
        D.y = scalbnf( dog.get( n.z, n.y+1, n.x ) - dog.get( n.z, n.y-1, n.x ), -1 );
        D.z = scalbnf( dog.get( n.z+1, n.y, n.x ) - dog.get( n.z-1, n.y, n.x ), -1 );

        const float c_val = dog.get( n.z, n.y, n.x );
        DD.x = dog.get( n.z, n.y, n.x+1 ) + dog.get( n.z, n.y, n.x-1 ) - scalbnf( c_val, 1 );
        DD.y = dog.get( n.z, n.y+1, n.x ) + dog.get( n.z, n.y-1, n.x ) - scalbnf( c_val, 1 );
        DD.z = dog.get( n.z+1, n.y, n.x ) + dog.get( n.z-1, n.y, n.x ) - scalbnf( c_val, 1 );

        DX.x = scalbnf( dog.get(n.z, n.y+1, n.x+1) + dog.get(n.z, n.y-1, n.x-1) - dog.get(n.z, n.y-1, n.x+1) - dog.get(n.z, n.y+1, n.x-1), -2 );
        DX.y = scalbnf( dog.get(n.z+1, n.y, n.x+1) + dog.get(n.z-1, n.y, n.x-1) - dog.get(n.z-1, n.y, n.x+1) - dog.get(n.z+1, n.y, n.x-1), -2 );
        DX.z = scalbnf( dog.get(n.z+1, n.y+1, n.x) + dog.get(n.z-1, n.y-1, n.x) - dog.get(n.z-1, n.y+1, n.x) - dog.get(n.z+1, n.y-1, n.x), -2 );

        // Build matrix for solve function
        float A[3][3];
        A[0][0] = DD.x; A[0][1] = DX.x; A[0][2] = DX.y;
        A[1][0] = DX.x; A[1][1] = DD.y; A[1][2] = DX.z;
        A[2][0] = DX.y; A[2][1] = DX.z; A[2][2] = DD.z;
        
        float3 b = {-D.x, -D.y, -D.z};
        
        if( ! solve( A, b ) ) {
            d.x = 0.0f;
            d.y = 0.0f;
            d.z = 0.0f;
            break;
        }
        
        d = b;  // solve() stores result in b

        const bool last_it = (iter == 4);
        int result = f.refine( d, n, width, height, maxlevel, last_it );
        if( result == 1 ) break;
    }

    if( d.x >= 1.5f || d.y >= 1.5f || d.z >= 1.5f ) {
        POP_INFO2( no_extrema_reporting, "Failed due to excessive repositioning" );
        return false;
    }

    const float xn      = n.x + d.x;
    const float yn      = n.y + d.y;
    const float sn      = n.z + d.z;

    if( ! verify( xn, yn, sn, width, height, maxlevel ) ) {
        POP_INFO2( no_extrema_reporting, "Failed due to optimum outside image plane" );
        return false;
    }

    const float contr   = v + scalbnf( D.x * d.x + D.y * d.y + D.z * d.z , -1 );
    const float tr      = DD.x + DD.y;
    const float det     = DD.x * DD.y - DX.x * DX.x;
    const float edgeval = tr * tr / det;

    if (det <= 0.0f) {
        POP_INFO2( no_extrema_reporting, "Failed due to saddle-shaped optimum" );
        return false;
    }

    if( fabsf(contr) < scalbnf( h_consts.threshold, 1 ) )
    {
        POP_INFO2( no_extrema_reporting, "Failed because contrast threshold exceeded" );
        return false;
    }

    if( edgeval >= (h_consts.edge_limit+1.0f)*(h_consts.edge_limit+1.0f)/h_consts.edge_limit ) {
        POP_INFO2( no_extrema_reporting, "Failed because edge threshold exceeded" );
        return false;
    }

    ec.xpos      = xn;
    ec.ypos      = yn;
    ec.lpos      = (int)roundf(sn);
    ec.sigma     = h_consts.sigma0 * pow(h_consts.sigma_k, sn);
    ec.cell      = floorf( yn / h_grid_divider ) * grid_width + floorf( xn / w_grid_divider );

    POP_INFO2( no_extrema_reporting, "Succeeded" );
    return true;
}

// template<int sift_mode>
// static
// void find_extrema_in_dog( const int3&    g,
//                           Plane2D_float& dog,
//                           int            octave,
//                           int            width,
//                           int            height,
//                           const uint32_t maxlevel,
//                           const float    w_grid_divider,
//                           const float    h_grid_divider,
//                           const int      grid_width )
// {
//     const bool no_extrema_reporting = false;

//     std::vector<InitialExtremum>& i_extrema = dct.initial_extrema_in_octave[octave];

//     POP_INFO2( no_extrema_reporting, "initial extrema values for octave " << octave );
//     for( int z=0; z<g.z; z++ )
//     {
//         for( int y=0; y<g.y; y++ )
//         {
//             for( int x=0; x<g.x; x++ )
//             {
//                 InitialExtremum ec;
//                 ec.ignore = false;

//                 int3 gi( x, y, z );

//                 bool indicator = find_extrema_in_dog_sub<sift_mode>( gi,
//                                                                      dog,
//                                                                      octave,
//                                                                      width,
//                                                                      height,
//                                                                      maxlevel,
//                                                                      w_grid_divider,
//                                                                      h_grid_divider,
//                                                                      grid_width,
//                                                                      ec );

//                 if( indicator )
//                 {
//                     // store the initial extremum in an array
//                     i_extrema.emplace_back( ec );
//                 }
//             }
//         }

//         std::vector<int>& i_ext_off = dct.initial_extrema_offset[octave];

//         i_ext_off.resize( i_extrema.size() );

//         for( int w_idx=0; w_idx<i_extrema.size(); w_idx++ )
//         {
//             i_extrema[w_idx].write_index = w_idx;
//             i_ext_off[w_idx]             = w_idx;
//         }


//         POP_INFO2( no_extrema_reporting, "Number of extrema in octave " << octave << " after level " << z << ": " << i_extrema.size() );
//     }

//     dct.extrema_count_per_octave[octave] = i_extrema.size();

//     POP_INFO2( no_extrema_reporting, "final extrema count in octave " << octave << ": " << dct.extrema_count_per_octave[octave] );
// }


struct ExtremaBuffer {
    InitialExtremum* extrema;
    int* count;
    int max_extrema;
};

template<int sift_mode>
static
void find_extrema_in_dog( const int3&    g,
                          Plane2D_float& dog,
                          int            octave,
                          int            width,
                          int            height,
                          const uint32_t maxlevel,
                          const float    w_grid_divider,
                          const float    h_grid_divider,
                          const int      grid_width,
                          Pyramid*       pyramid )
{
    const bool no_extrema_reporting = false;

    std::vector<InitialExtremum>& i_extrema = dct.initial_extrema_in_octave[octave];

    POP_INFO2( no_extrema_reporting, "Converting find_extrema to SYCL kernel for octave " << octave );

    Octave& oct_obj = pyramid->getOctave(octave);
    sycl::queue& queue = oct_obj.getQueue();

    float* dog_ptr = (float*)dog.getDevicePtr();
    const int dog_pitch = dog.getPitchElements();

    // DEBUG: Copy first layer of DoG to host to verify data
    const int layer0_size = dog_pitch * height;
    std::vector<float> host_dog_layer0(layer0_size);
    queue.memcpy(host_dog_layer0.data(), dog_ptr, layer0_size * sizeof(float)).wait();
    
    // POP_INFO2( false, "DEBUG: First 10 DoG values in octave " << octave << ": ");
    // for(int i = 0; i < std::min(10, layer0_size); i++) {
    //     std::cout << host_dog_layer0[i] << " ";
    // }
    // std::cout << std::endl;
    
    // Count non-zero values
    int non_zero_count = 0;
    for(int i = 0; i < layer0_size; i++) {
        if(std::abs(host_dog_layer0[i]) > 1e-6f) non_zero_count++;
    }
    POP_INFO2( false, "DEBUG: Non-zero DoG values: " << non_zero_count << " / " << layer0_size );

    const int max_extrema = g.x * g.y * g.z;
    InitialExtremum* d_extrema = sycl::malloc_device<InitialExtremum>(max_extrema, queue);
    int* d_count = sycl::malloc_device<int>(1, queue);
    
    queue.memset(d_count, 0, sizeof(int)).wait();

    const float threshold = h_consts.threshold;
    const float edge_limit = h_consts.edge_limit;
    const float sigma0 = h_consts.sigma0;
    const float sigma_k = h_consts.sigma_k;

    POP_INFO2( no_extrema_reporting, "Launching SYCL kernel for extrema detection: " 
               << g.x << "x" << g.y << "x" << g.z );
    POP_INFO2( false, "DEBUG: dog_pitch=" << dog_pitch << ", threshold=" << threshold << ", edge_limit=" << edge_limit );

    // Launch SYCL kernel
    auto event = queue.submit([&](sycl::handler& cgh) {
        const int c_width = width;
        const int c_height = height;
        const int c_maxlevel = maxlevel;
        const int c_dog_pitch = dog_pitch;
        const int c_grid_width = grid_width;
        const float c_w_grid_div = w_grid_divider;
        const float c_h_grid_div = h_grid_divider;
        const float c_threshold = threshold;
        const float c_edge_limit = edge_limit;
        const float c_sigma0 = sigma0;
        const float c_sigma_k = sigma_k;

    cgh.parallel_for(
        sycl::range<3>(g.z, g.y, g.x),
        [=](sycl::id<3> idx) {
            // Grid coordinates (0-based)
            const int gx = idx[2];
            const int gy = idx[1];
            const int gz = idx[0];
            
            // Actual coordinates for refinement (1-based, with border)
            const int x = gx + 1;
            const int y = gy + 1;
            const int level = gz + 1;

            auto dog_get = [=](int z, int yi, int xi) -> float {
                const int dog_idx = (z * c_height + yi) * c_dog_pitch + xi;
                return dog_ptr[dog_idx];
            };

            // The value to check is at the actual grid position
            // In CPU: is_extremum(dog, x-1, y-1, level-1) with TX(1,1,1)
            // gives dog.get((level-1)+1, (y-1)+1, (x-1)+1) = dog.get(level, y, x)
            const float val = dog_get(level, y, x);

            // First contrast check
            if (sycl::fabs(val) < 1.6f * c_threshold) return;

            // Now check 26 neighbors around (level, y, x)
            // The CPU TX macro adds to (x-1, y-1, level-1), so:
            // TX(0,1,1) = dog.get((level-1)+1, (y-1)+1, (x-1)+0) = dog.get(level, y, x-1)
            // TX(2,1,1) = dog.get((level-1)+1, (y-1)+1, (x-1)+2) = dog.get(level, y, x+1)
            
            uint32_t gt = 0;
            uint32_t lt = 0;

            auto extremum_cmp = [&](float f, uint32_t mask) {
                gt |= ((val > f) ? mask : 0);
                lt |= ((val < f) ? mask : 0);
            };

            // 1st group: TX(0,1,1) and TX(2,1,1)
            extremum_cmp(dog_get(level, y, x - 1), 0x00400000);
            extremum_cmp(dog_get(level, y, x + 1), 0x00040000);
            
            if((gt != 0x00440000) && (lt != 0x00440000)) return;

            // 2nd group: TX(1,0,1), TX(1,2,1), TX(1,0,0), TX(1,2,0), TX(1,1,0), TX(1,0,2), TX(1,1,2), TX(1,2,2)
            extremum_cmp(dog_get(level, y - 1, x), 0x00800000);
            extremum_cmp(dog_get(level, y + 1, x), 0x00200000);
            extremum_cmp(dog_get(level - 1, y - 1, x), 0x80000000);
            extremum_cmp(dog_get(level - 1, y + 1, x), 0x40000000);
            extremum_cmp(dog_get(level - 1, y, x), 0x20000000);
            extremum_cmp(dog_get(level + 1, y - 1, x), 0x00008000);
            extremum_cmp(dog_get(level + 1, y, x), 0x00004000);
            extremum_cmp(dog_get(level + 1, y + 1, x), 0x00002000);

            if((gt != 0xe0e4e000) && (lt != 0xe0e4e000)) return;

            // 3rd group: TX(0,0,1), TX(2,0,1), TX(0,2,1), TX(2,2,1)
            extremum_cmp(dog_get(level, y - 1, x - 1), 0x00010000);
            extremum_cmp(dog_get(level, y - 1, x + 1), 0x00020000);
            extremum_cmp(dog_get(level, y + 1, x - 1), 0x00100000);
            extremum_cmp(dog_get(level, y + 1, x + 1), 0x00080000);

            if((gt != 0xe0ffe000) && (lt != 0xe0ffe000)) return;

            // 4th group: TX(0,0,0), TX(2,0,0), TX(0,1,0), TX(2,1,0), TX(0,2,0), TX(2,2,0)
            extremum_cmp(dog_get(level - 1, y - 1, x - 1), 0x01000000);
            extremum_cmp(dog_get(level - 1, y - 1, x + 1), 0x02000000);
            extremum_cmp(dog_get(level - 1, y, x - 1), 0x00000004);
            extremum_cmp(dog_get(level - 1, y, x + 1), 0x04000000);
            extremum_cmp(dog_get(level - 1, y + 1, x - 1), 0x10000000);
            extremum_cmp(dog_get(level - 1, y + 1, x + 1), 0x08000000);

            if((gt != 0xffffe004) && (lt != 0xffffe004)) return;

            // 5th group: TX(0,0,2), TX(2,0,2), TX(0,1,2), TX(2,1,2), TX(0,2,2), TX(2,2,2)
            extremum_cmp(dog_get(level + 1, y - 1, x - 1), 0x00000100);
            extremum_cmp(dog_get(level + 1, y - 1, x + 1), 0x00000200);
            extremum_cmp(dog_get(level + 1, y, x - 1), 0x00000001);
            extremum_cmp(dog_get(level + 1, y, x + 1), 0x00000400);
            extremum_cmp(dog_get(level + 1, y + 1, x - 1), 0x00001000);
            extremum_cmp(dog_get(level + 1, y + 1, x + 1), 0x00000800);

            if((gt != 0xffffff05) && (lt != 0xffffff05)) return;

            // NOW the extremum check passed, use the SAME value v for refinement
            const float v = val;  // This is already dog_get(level, y, x)



                //const float v = dog_get(level, y, x);

                // Refinement loop - use coordinates (x, y, level) NOT (check_x, check_y, check_z)
                float3 d = {0.0f, 0.0f, 0.0f};
                int3 n = {x, y, level};
                int iter = 0;
                const int MAX_ITER = 5;

                while(iter < MAX_ITER) {
                    iter++;

                    // ... gradient and Hessian computation stays the same ...
                    
                    const float x2y1z1 = dog_get(n.z, n.y, n.x + 1);
                    const float x0y1z1 = dog_get(n.z, n.y, n.x - 1);
                    const float x1y2z1 = dog_get(n.z, n.y + 1, n.x);
                    const float x1y0z1 = dog_get(n.z, n.y - 1, n.x);
                    const float x1y1z2 = dog_get(n.z + 1, n.y, n.x);
                    const float x1y1z0 = dog_get(n.z - 1, n.y, n.x);

                    float3 D;
                    D.x = scalbnf(x2y1z1 - x0y1z1, -1);
                    D.y = scalbnf(x1y2z1 - x1y0z1, -1);
                    D.z = scalbnf(x1y1z2 - x1y1z0, -1);

                    const float x1y1z1 = dog_get(n.z, n.y, n.x);
                    float3 DD;
                    DD.x = x2y1z1 + x0y1z1 - scalbnf(x1y1z1, 1);
                    DD.y = x1y2z1 + x1y0z1 - scalbnf(x1y1z1, 1);
                    DD.z = x1y1z2 + x1y1z0 - scalbnf(x1y1z1, 1);

                    const float x0y0z1 = dog_get(n.z, n.y - 1, n.x - 1);
                    const float x0y2z1 = dog_get(n.z, n.y + 1, n.x - 1);
                    const float x2y0z1 = dog_get(n.z, n.y - 1, n.x + 1);
                    const float x2y2z1 = dog_get(n.z, n.y + 1, n.x + 1);
                    const float x0y1z0 = dog_get(n.z - 1, n.y, n.x - 1);
                    const float x0y1z2 = dog_get(n.z + 1, n.y, n.x - 1);
                    const float x2y1z0 = dog_get(n.z - 1, n.y, n.x + 1);
                    const float x2y1z2 = dog_get(n.z + 1, n.y, n.x + 1);
                    const float x1y0z0 = dog_get(n.z - 1, n.y - 1, n.x);
                    const float x1y0z2 = dog_get(n.z + 1, n.y - 1, n.x);
                    const float x1y2z0 = dog_get(n.z - 1, n.y + 1, n.x);
                    const float x1y2z2 = dog_get(n.z + 1, n.y + 1, n.x);

                    float3 DX;
                    DX.x = scalbnf(x2y2z1 + x0y0z1 - x0y2z1 - x2y0z1, -2);
                    DX.y = scalbnf(x2y1z2 + x0y1z0 - x0y1z2 - x2y1z0, -2);
                    DX.z = scalbnf(x1y2z2 + x1y0z0 - x1y2z0 - x1y0z2, -2);

                    float A[3][3];
                    A[0][0] = DD.x; A[0][1] = DX.x; A[0][2] = DX.y;
                    A[1][0] = DX.x; A[1][1] = DD.y; A[1][2] = DX.z;
                    A[2][0] = DX.y; A[2][1] = DX.z; A[2][2] = DD.z;

                    float3 b = {-D.x, -D.y, -D.z};

                    // Compute determinants for matrix inversion (matching s_solve.h)
                    float det0b = -A[1][2] * A[1][2];
                    float det0a = A[1][1] * A[2][2];
                    float det0 = det0b + det0a;

                    float det1b = -A[0][1] * A[2][2];
                    float det1a = A[1][2] * A[0][2];
                    float det1 = det1b + det1a;

                    float det2b = -A[1][1] * A[0][2];
                    float det2a = A[0][1] * A[1][2];
                    float det2 = det2b + det2a;

                    float det3b = -A[0][2] * A[0][2];
                    float det3a = A[0][0] * A[2][2];
                    float det3 = det3b + det3a;

                    float det4b = -A[0][0] * A[1][2];
                    float det4a = A[0][1] * A[0][2];
                    float det4 = det4b + det4a;

                    float det5b = -A[0][1] * A[0][1];
                    float det5a = A[0][0] * A[1][1];
                    float det5 = det5b + det5a;

                    float det = (A[0][0] * det0) + (A[0][1] * det1) + (A[0][2] * det2);

                    if(sycl::fabs(det) < 1e-10f) {
                        d.x = 0.0f;
                        d.y = 0.0f;
                        d.z = 0.0f;
                        break;
                    }

                    float rsd = 1.0f / det;

                    // Compute inverse matrix
                    float inv[3][3];
                    inv[0][0] = det0 * rsd;
                    inv[1][0] = det1 * rsd;
                    inv[2][0] = det2 * rsd;
                    inv[1][1] = det3 * rsd;
                    inv[1][2] = det4 * rsd;
                    inv[2][2] = det5 * rsd;
                    inv[0][1] = inv[1][0];
                    inv[0][2] = inv[2][0];
                    inv[2][1] = inv[1][2];

                    // Multiply inv * b to get solution
                    d.x = inv[0][0] * b.x + inv[0][1] * b.y + inv[0][2] * b.z;
                    d.y = inv[1][0] * b.x + inv[1][1] * b.y + inv[1][2] * b.z;
                    d.z = inv[2][0] * b.x + inv[2][1] * b.y + inv[2][2] * b.z;

                    // Match CPU refine logic: on last iteration, don't check for movement
                    const bool last_it = (iter == MAX_ITER);
                    if(last_it) break;  // CPU returns 0 (continue), but loop ends anyway
                    
                    // Check if refinement requires position change
                    int3 t = {0, 0, 0};
                    
                    t.x = ((d.x >= 0.6f && n.x < c_width - 2) ? 1 : 0) +
                        ((d.x <= -0.6f && n.x > 1) ? -1 : 0);
                    t.y = ((d.y >= 0.6f && n.y < c_height - 2) ? 1 : 0) +
                        ((d.y <= -0.6f && n.y > 1) ? -1 : 0);
                    
                    if constexpr (sift_mode == Config::RefineInOctave) {
                        t.z = ((d.z >= 0.6f && n.z < c_maxlevel - 1) ? 1 : 0) +
                            ((d.z <= -0.6f && n.z > 1) ? -1 : 0);
                    }
                    
                    if(t.x == 0 && t.y == 0 && t.z == 0) break;  // No movement, converged
                    
                    n.x += t.x;
                    n.y += t.y;
                    n.z += t.z;
                }

                // Final validation
                if(sycl::fabs(d.x) >= 1.5f || sycl::fabs(d.y) >= 1.5f || sycl::fabs(d.z) >= 1.5f) return;

                const float xn = n.x + d.x;
                const float yn = n.y + d.y;
                const float sn = n.z + d.z;


                if(xn < 0.0f || xn > c_width - 1.0f ||   // Match CPU: 0 <= xn <= 63
                   yn < 0.0f || yn > c_height - 1.0f ||  // Match CPU: 0 <= yn <= 63
                   sn < 0.5f || sn > c_maxlevel -0.5f) return;

                // Use v (the original value at x,y,level) and D (from last iteration) for contrast
                // This matches CPU line 283: const float contr = v + 0.5f * (D.x * d.x + D.y * d.y + D.z * d.z);
                // Get the LAST computed D values (they're still in scope from last iteration)
                
                // Actually, we need to recompute D at final position to match CPU
                const float x2y1z1_f = dog_get(n.z, n.y, n.x + 1);
                const float x0y1z1_f = dog_get(n.z, n.y, n.x - 1);
                const float x1y2z1_f = dog_get(n.z, n.y + 1, n.x);
                const float x1y0z1_f = dog_get(n.z, n.y - 1, n.x);
                const float x1y1z2_f = dog_get(n.z + 1, n.y, n.x);
                const float x1y1z0_f = dog_get(n.z - 1, n.y, n.x);
                const float x1y1z1_f = dog_get(n.z, n.y, n.x);

                float3 D_f;
                D_f.x = scalbnf(x2y1z1_f - x0y1z1_f, -1);
                D_f.y = scalbnf(x1y2z1_f - x1y0z1_f, -1);
                D_f.z = scalbnf(x1y1z2_f - x1y1z0_f, -1);

                float3 DD_f;
                DD_f.x = x2y1z1_f + x0y1z1_f - scalbnf(x1y1z1_f, 1);
                DD_f.y = x1y2z1_f + x1y0z1_f - scalbnf(x1y1z1_f, 1);

                const float contr = v + scalbnf(D_f.x * d.x + D_f.y * d.y + D_f.z * d.z, -1);
                
                if(sycl::fabs(contr) < scalbnf(c_threshold, 1)) return;  // Instead of c_threshold * 2.0f

                const float tr = DD_f.x + DD_f.y;
                const float x0y0z1_f = dog_get(n.z, n.y - 1, n.x - 1);
                const float x0y2z1_f = dog_get(n.z, n.y + 1, n.x - 1);
                const float x2y0z1_f = dog_get(n.z, n.y - 1, n.x + 1);
                const float x2y2z1_f = dog_get(n.z, n.y + 1, n.x + 1);
                const float DXx_f = scalbnf(x2y2z1_f + x0y0z1_f - x0y2z1_f - x2y0z1_f, -2);
                const float det_f = DD_f.x * DD_f.y - DXx_f * DXx_f;
                if(det_f <= 0.0f) return;

                const float edgeval = tr * tr / det_f;
                if(edgeval >= (c_edge_limit + 1.0f) * (c_edge_limit + 1.0f) / c_edge_limit) return;

                // Atomically add extremum
                int write_idx = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(d_count[0]).fetch_add(1);
                
                if(write_idx < max_extrema) {
                    InitialExtremum& ec = d_extrema[write_idx];
                    ec.xpos = xn;
                    ec.ypos = yn;
                    ec.lpos = (int)sycl::round(sn);
                    ec.sigma = c_sigma0 * sycl::pow(c_sigma_k, sn);
                    ec.cell = sycl::floor(yn / c_h_grid_div) * c_grid_width + sycl::floor(xn / c_w_grid_div);
                    ec.ignore = false;
                    ec.write_index = write_idx;
                }
            }
        );
    });

    event.wait();

    // Copy results back to host
    int extrema_count = 0;
    queue.memcpy(&extrema_count, d_count, sizeof(int)).wait();

    POP_INFO2( no_extrema_reporting, "Found " << extrema_count << " extrema on device" );

    if(extrema_count > 0) {
                std::vector<InitialExtremum> host_extrema(extrema_count);
                queue.memcpy(host_extrema.data(), d_extrema, extrema_count * sizeof(InitialExtremum)).wait();
                
                // Deduplicate extrema that are too close (within 0.001 pixels and same level)
                std::vector<InitialExtremum> unique_extrema;
                for(const auto& ex : host_extrema) {
                    bool is_duplicate = false;
                    for(const auto& existing : unique_extrema) {
                        if(existing.lpos == ex.lpos &&
                        std::abs(existing.xpos - ex.xpos) < 0.001f &&
                        std::abs(existing.ypos - ex.ypos) < 0.001f) {
                            is_duplicate = true;
                            break;
                        }
                    }
                    if(!is_duplicate) {
                        unique_extrema.push_back(ex);
                    }
                }
                
                POP_INFO2(false, "After deduplication: " << unique_extrema.size() << " / " << extrema_count << " extrema");
                
                // DEBUG: Print extrema positions grouped by level (like CPU version)
                POP_INFO2( false, "Extrema in octave " << octave << " by level:" );
                
                // Group by level (lpos)
                std::map<int, std::vector<InitialExtremum>> by_level;
                for(const auto& ex : unique_extrema) {
                    by_level[ex.lpos].push_back(ex);
                }
                
                // for(const auto& [level, extrema_in_level] : by_level) {
                //     POP_INFO2( false, "  Level " << level << " (" << extrema_in_level.size() << " extrema):" );
                //     for(const auto& ex : extrema_in_level) {
                //         POP_INFO2( false, "    pos=(" << ex.xpos << ", " << ex.ypos << ", " << ex.lpos 
                //                 << ") sigma=" << ex.sigma );
                //     }
                // }
                
                i_extrema.insert(i_extrema.end(), unique_extrema.begin(), unique_extrema.end());
            }

    // Cleanup
    sycl::free(d_extrema, queue);
    sycl::free(d_count, queue);

    dct.extrema_count_per_octave[octave] = i_extrema.size();

    POP_INFO2( no_extrema_reporting, "final extrema count in octave " << octave << ": " << dct.extrema_count_per_octave[octave] );
}



void Pyramid::find_extrema( const Config& conf )
{
    POP_INFO2( false, "Enter " << __FUNCTION__ );

    dct.extrema_count_per_octave.resize( MAX_OCTAVES );

    for( int octave=0; octave<_num_octaves; octave++ )
    {
        Octave& oct_obj = _octaves[octave];

        int* extrema_num_blocks = getNumberOfBlocks( octave );

        int cols = oct_obj.getWidth();
        int rows = oct_obj.getHeight();

        switch( conf.getSiftMode() )
        {
        case Config::RefineInLevel :
                find_extrema_in_dog<Config::RefineInLevel>
                    ( int3(cols, rows, _levels-3),
                      oct_obj.getDog( ),
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize(),
                      this );  // ADD this parameter
                break;
        default :
                find_extrema_in_dog<Config::RefineInOctave>
                    ( int3(cols, rows, _levels-3),
                      oct_obj.getDog( ),
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize(),
                      this );  // ADD this parameter
                break;
        }

        bool log_to_file = ( conf.getLogMode() == popsift::Config::All );
        if( log_to_file ) {
            struct stat st = { 0 };

            if (stat("dir-extrema", &st) == -1) {
                mkdir("dir-extrema", 0700);
            }

            std::vector<int2> red_pixel_list;
            for( auto it : dct.initial_extrema_in_octave[octave] )
            {
                red_pixel_list.emplace_back( int2( roundf(it.xpos), roundf(it.ypos) ) );
            }

            std::ostringstream ostr;
            ostr << "dir-extrema/" << "pyramid" << "-o-" << octave << "-red" << ".ppm";
            popsift::write_plane2Dppm( ostr.str().c_str(), oct_obj.getData(), red_pixel_list );
        }
    }

    POP_INFO2( false, "found extrema in all octaves" );
 
    /* Copy the extreme count for every octave from the (already initialized)
     * array extrema_count_per_octave to the (uninitialized) array extrema_count_prefix_sum. */
    dct.extrema_count_prefix_sum.resize( dct.extrema_count_per_octave.size() + 1 );

    auto it = dct.extrema_count_prefix_sum.begin();
    *it = 0;
    it++;

    /* Compute the exclusive prefix sum on the array extrema_count_prefix_sum, but add
     * the total sum in the last element. Easier to achieve with an inclusive_scan. */
    std::inclusive_scan( dct.extrema_count_per_octave.begin(),
                         dct.extrema_count_per_octave.end(),
                         it );

    /* Store the total number of orientations and the total number of
     * extrema as well. */
    dct.extrema_count_total = dct.extrema_count_prefix_sum.back();

    std::ostringstream debug_ostr;
    debug_ostr << "Extrema per octave:" << std::endl;
    std::copy( dct.extrema_count_per_octave.begin(),
               dct.extrema_count_per_octave.end(),
               std::ostream_iterator<int>(debug_ostr, " ") );
    debug_ostr << std::endl
          << "Extrema prefix sum per octave:" << std::endl;
    std::copy( dct.extrema_count_prefix_sum.begin(),
               dct.extrema_count_prefix_sum.end(),
               std::ostream_iterator<int>(debug_ostr, " ") );
    debug_ostr << std::endl
          << "Extrema prefix sum per octave: "
          << dct.extrema_count_total
          << std::endl;
    POP_INFO2( false, debug_ostr.str() );
}

} // namespace popsift

