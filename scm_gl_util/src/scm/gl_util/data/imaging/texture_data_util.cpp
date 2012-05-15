
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_data_util.h"

#include <memory.h>

#include <scm/gl_core/log.h>
#include <scm/gl_core/texture_objects/texture_image.h>
#include <scm/gl_util/data/imaging/mip_map_generation.h>

namespace scm {
namespace gl {
namespace util {

namespace nv {
// thanks nvidia for providing the source code to flip dxt images
typedef struct
{
    unsigned short col0, col1;
    unsigned char row[4];
} DXTColorBlock_t;

typedef struct
{
    unsigned short row[4];
} DXT3AlphaBlock_t;

typedef struct
{
    unsigned char alpha0, alpha1;
    unsigned char row[6];
} DXT5AlphaBlock_t;

inline
void
SwapMem(void *byte1, void *byte2, int size)
{
    unsigned char *tmp=(unsigned char *)malloc(sizeof(unsigned char)*size);
    memcpy(tmp, byte1, size);
    memcpy(byte1, byte2, size);
    memcpy(byte2, tmp, size);
    free(tmp);
}

inline
void
SwapChar(unsigned char * x, unsigned char * y)
{
    unsigned char z = *x;
    *x = *y;
    *y = z;
}

inline
void
SwapShort(unsigned short * x, unsigned short * y)
{
    unsigned short z = *x;
    *x = *y;
    *y = z;
}

inline
void
flipDXT1Blocks(DXTColorBlock_t *Block, int NumBlocks)
{
    int i;
    DXTColorBlock_t *ColorBlock=Block;
    for(i=0; i<NumBlocks; ++i) {
        SwapChar( &ColorBlock->row[0], &ColorBlock->row[3] );
        SwapChar( &ColorBlock->row[1], &ColorBlock->row[2] );
        ++ColorBlock;
    }
}

inline
void
flipDXT3Blocks(DXTColorBlock_t *Block, int NumBlocks)
{
    int i;
    DXTColorBlock_t *ColorBlock=Block;
    DXT3AlphaBlock_t *AlphaBlock;
    for(i=0; i<NumBlocks; ++i) {
        AlphaBlock=(DXT3AlphaBlock_t *)ColorBlock;
        SwapShort( &AlphaBlock->row[0], &AlphaBlock->row[3] );
        SwapShort( &AlphaBlock->row[1], &AlphaBlock->row[2] );
        ++ColorBlock;
        SwapChar( &ColorBlock->row[0], &ColorBlock->row[3] );
        SwapChar( &ColorBlock->row[1], &ColorBlock->row[2] );
        ++ColorBlock;
    }
}

inline
void
flipDXT5Alpha(DXT5AlphaBlock_t *Block)
{
    unsigned long *Bits, Bits0=0, Bits1=0;

    memcpy(&Bits0, &Block->row[0], sizeof(unsigned char)*3);
    memcpy(&Bits1, &Block->row[3], sizeof(unsigned char)*3);

    Bits=((unsigned long *)&(Block->row[0]));
    *Bits&=0xff000000;
    *Bits|=(unsigned char)(Bits1>>12)&0x00000007;
    *Bits|=(unsigned char)((Bits1>>15)&0x00000007)<<3;
    *Bits|=(unsigned char)((Bits1>>18)&0x00000007)<<6;
    *Bits|=(unsigned char)((Bits1>>21)&0x00000007)<<9;
    *Bits|=(unsigned char)(Bits1&0x00000007)<<12;
    *Bits|=(unsigned char)((Bits1>>3)&0x00000007)<<15;
    *Bits|=(unsigned char)((Bits1>>6)&0x00000007)<<18;
    *Bits|=(unsigned char)((Bits1>>9)&0x00000007)<<21;

    Bits=((unsigned long *)&(Block->row[3]));
    *Bits&=0xff000000;
    *Bits|=(unsigned char)(Bits0>>12)&0x00000007;
    *Bits|=(unsigned char)((Bits0>>15)&0x00000007)<<3;
    *Bits|=(unsigned char)((Bits0>>18)&0x00000007)<<6;
    *Bits|=(unsigned char)((Bits0>>21)&0x00000007)<<9;
    *Bits|=(unsigned char)(Bits0&0x00000007)<<12;
    *Bits|=(unsigned char)((Bits0>>3)&0x00000007)<<15;
    *Bits|=(unsigned char)((Bits0>>6)&0x00000007)<<18;
    *Bits|=(unsigned char)((Bits0>>9)&0x00000007)<<21;
}

inline
void
flipDXT5Blocks(DXTColorBlock_t *Block, int NumBlocks)
{
    DXTColorBlock_t *ColorBlock=Block;
    DXT5AlphaBlock_t *AlphaBlock;
    int i;

    for(i=0; i<NumBlocks; ++i) {
        AlphaBlock=(DXT5AlphaBlock_t *)ColorBlock;

        flipDXT5Alpha(AlphaBlock);
        ++ColorBlock;

        SwapChar( &ColorBlock->row[0], &ColorBlock->row[3] );
        SwapChar( &ColorBlock->row[1], &ColorBlock->row[2] );
        ++ColorBlock;
    }
}

inline
void
flipBC4Blocks(DXTColorBlock_t *Block, int NumBlocks)
{
    DXTColorBlock_t  *ColorBlock = Block;
    DXT5AlphaBlock_t *RedBlock;

    for(int i = 0; i < NumBlocks; ++i) {
        RedBlock = reinterpret_cast<DXT5AlphaBlock_t*>(ColorBlock);

        flipDXT5Alpha(RedBlock);
        ++ColorBlock;
    }
}

inline
void
flipBC5Blocks(DXTColorBlock_t *Block, int NumBlocks)
{
    DXTColorBlock_t  *ColorBlock = Block;
    DXT5AlphaBlock_t *RBlock;
    DXT5AlphaBlock_t *GBlock;

    for(int i = 0; i < NumBlocks; ++i) {
        RBlock = reinterpret_cast<DXT5AlphaBlock_t*>(ColorBlock);
        flipDXT5Alpha(RBlock);
        ++ColorBlock;

        GBlock = reinterpret_cast<DXT5AlphaBlock_t*>(ColorBlock);
        flipDXT5Alpha(GBlock);
        ++ColorBlock;
    }
}

} // namespace nv

bool
image_layer_vert_flip_raw(scm::uint8*const data,
                          data_format      fmt,
                          unsigned         w,
                          unsigned         h)
{
    if (is_compressed_format(fmt)) {
        using namespace scm::gl::util::nv;

        int bw  = (w + 3) / 4;
        int bh  = (h + 3) / 4;
        int bs  = compressed_block_size(fmt);
        int bls = bw * bs;

        if (1 == h) {
            return true;
        }
        else if (1 < h && h < 4) {
            DXTColorBlock_t *line = reinterpret_cast<DXTColorBlock_t*>(data);
            switch (fmt) {
                case FORMAT_BC1_RGBA:
                case FORMAT_BC1_SRGBA:
                    flipDXT1Blocks(line, bw);
                    break;
                case FORMAT_BC2_RGBA:
                case FORMAT_BC2_SRGBA:
                    flipDXT3Blocks(line, bw);
                    break;
                case FORMAT_BC3_RGBA:
                case FORMAT_BC3_SRGBA:
                    flipDXT5Blocks(line, bw);
                    break;
                case FORMAT_BC4_R:
                case FORMAT_BC4_R_S:
                    flipBC4Blocks(line, bw);
                    break;
                case FORMAT_BC5_RG:
                case FORMAT_BC5_RG_S:
                    flipBC5Blocks(line, bw);
                    break;
                default:
                    return false;
            }
            return true;
        }
        else if (4 <= h) {
            DXTColorBlock_t *top_line;
            DXTColorBlock_t *bottom_line;

            switch (fmt) {
                case FORMAT_BC1_RGBA:
                case FORMAT_BC1_SRGBA:
                    for(int bl = 0; bl < (bh / 2); ++bl) {
                        top_line    = reinterpret_cast<DXTColorBlock_t*>(data + bl * bls);
                        bottom_line = reinterpret_cast<DXTColorBlock_t*>(data + (bh - (bl + 1)) * bls);
                        flipDXT1Blocks(top_line,    bw);
                        flipDXT1Blocks(bottom_line, bw);
                        SwapMem(bottom_line, top_line, bls);
                    }
                    break;
                case FORMAT_BC2_RGBA:
                case FORMAT_BC2_SRGBA:
                    for(int bl = 0; bl < (bh / 2); ++bl) {
                        top_line    = reinterpret_cast<DXTColorBlock_t*>(data + bl * bls);
                        bottom_line = reinterpret_cast<DXTColorBlock_t*>(data + (bh - (bl + 1)) * bls);
                        flipDXT3Blocks(top_line,    bw);
                        flipDXT3Blocks(bottom_line, bw);
                        SwapMem(bottom_line, top_line, bls);
                    }
                    break;
                case FORMAT_BC3_RGBA:
                case FORMAT_BC3_SRGBA:
                    for(int bl = 0; bl < (bh / 2); ++bl) {
                        top_line    = reinterpret_cast<DXTColorBlock_t*>(data + bl * bls);
                        bottom_line = reinterpret_cast<DXTColorBlock_t*>(data + (bh - (bl + 1)) * bls);
                        flipDXT5Blocks(top_line,    bw);
                        flipDXT5Blocks(bottom_line, bw);
                        SwapMem(bottom_line, top_line, bls);
                    }
                    break;
                case FORMAT_BC4_R:
                case FORMAT_BC4_R_S:
                    for(int bl = 0; bl < (bh / 2); ++bl) {
                        top_line    = reinterpret_cast<DXTColorBlock_t*>(data + bl * bls);
                        bottom_line = reinterpret_cast<DXTColorBlock_t*>(data + (bh - (bl + 1)) * bls);
                        flipBC4Blocks(top_line,    bw);
                        flipBC4Blocks(bottom_line, bw);
                        SwapMem(bottom_line, top_line, bls);
                    }
                    break;
                case FORMAT_BC5_RG:
                case FORMAT_BC5_RG_S:
                    for(int bl = 0; bl < (bh / 2); ++bl) {
                        top_line    = reinterpret_cast<DXTColorBlock_t*>(data + bl * bls);
                        bottom_line = reinterpret_cast<DXTColorBlock_t*>(data + (bh - (bl + 1)) * bls);
                        flipBC5Blocks(top_line,    bw);
                        flipBC5Blocks(bottom_line, bw);
                        SwapMem(bottom_line, top_line, bls);
                    }
                    break;
                default:
                    return false;
            }
            return true;
        }
        else {
            return false;
        }
    }
    else {
        size_t              lsize = static_cast<size_t>(w) * size_of_format(fmt);
        scoped_array<uint8> tmp_line(new uint8[lsize]);

        for (unsigned l = 0; l < (h / 2); ++l) {
            size_t r = lsize * l;
            size_t w = lsize * (h - (l + 1));

            memcpy(tmp_line.get(), data + r,       lsize);
            memcpy(data + r,       data + w,       lsize);
            memcpy(data + w,       tmp_line.get(), lsize);
        }
        return true;
    }
}

bool
image_flip_vertical(const shared_array<uint8>& data,
                          data_format          fmt,
                          unsigned             w,
                          unsigned             h)
{
    return image_layer_vert_flip_raw(data.get(), fmt, w, h);
}

bool
volume_flip_vertical(const shared_array<uint8>& data,
                           data_format          fmt,
                           unsigned             w,
                           unsigned             h,
                           unsigned             d)
{
    size_t ssize = (static_cast<size_t>(w) * h * bit_per_pixel(fmt)) / 8;

    for (unsigned s = 0; s < d; ++s) {
        if (!image_layer_vert_flip_raw(data.get() + ssize * s, fmt, w, h)) {
            return false;
        }
    }

    return true;
}


bool
generate_mipmaps(const math::vec3ui&        src_dim,
                       gl::data_format      src_fmt,
                       uint8*               src_data,
                       std::vector<uint8*>& dst_data)
{
    using namespace scm::gl;
    using namespace scm::math;

    switch (src_fmt) {
    case FORMAT_R_32F:
        typed_generate_mipmaps<float, 1, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_RG_32F:
        typed_generate_mipmaps<float, 2, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_RGB_32F:
        typed_generate_mipmaps<float, 3, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_RGBA_32F:
        typed_generate_mipmaps<float, 4, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_R_8:
        typed_generate_mipmaps<uint8, 1, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_RG_8:
        typed_generate_mipmaps<uint8, 2, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_RGB_8:
        typed_generate_mipmaps<uint8, 3, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_RGBA_8:
        typed_generate_mipmaps<uint8, 4, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_R_16:
        typed_generate_mipmaps<uint16, 1, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_RG_16:
        typed_generate_mipmaps<uint16, 2, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_RGB_16:
        typed_generate_mipmaps<uint16, 3, 2>(src_dim, src_data, dst_data);
        break;
    case FORMAT_RGBA_16:
        typed_generate_mipmaps<uint16, 4, 2>(src_dim, src_data, dst_data);
        break;
    default:
        glerr() << log::error
                << "generate_mipmaps(): error unsupported source data format (" << format_string(src_fmt) << ")." << log::end;
        return false;
    }

    return true;
}

} // namespace util
} // namespace gl
} // namespace scm
