
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_loader_dds.h"

#include <cassert>
#include <exception>
#include <stdexcept>
#include <string>

#include <boost/static_assert.hpp>

#include <scm/core/numeric_types.h>
#include <scm/core/math.h>
#include <scm/core/memory.h>
#include <scm/core/io/file.h>

#include <scm/gl_core/log.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/data/imaging/texture_image_data.h>
#include <scm/gl_util/data/imaging/texture_data_util.h>

#define SCM_MAKEFOURCC(ch0, ch1, ch2, ch3)                          \
    (scm::uint32)(                                                  \
    (((scm::uint32)(scm::uint8)(ch3) << 24) & 0xFF000000) |         \
    (((scm::uint32)(scm::uint8)(ch2) << 16) & 0x00FF0000) |         \
    (((scm::uint32)(scm::uint8)(ch1) <<  8) & 0x0000FF00) |         \
     ((scm::uint32)(scm::uint8)(ch0)        & 0x000000FF) )

#define DDS_MAGIC 0x20534444 // "DDS "

//  DDS_HEADER.dwFlags
#define DDSD_CAPS                   0x00000001
#define DDSD_HEIGHT                 0x00000002
#define DDSD_WIDTH                  0x00000004
#define DDSD_PITCH                  0x00000008
#define DDSD_PIXELFORMAT            0x00001000
#define DDSD_MIPMAPCOUNT            0x00020000
#define DDSD_LINEARSIZE             0x00080000
#define DDSD_DEPTH                  0x00800000

//  DDS_HEADER.sPixelFormat.dwFlags
#define DDPF_ALPHAPIXELS            0x00000001
#define DDPF_ALPHA                  0x00000002
#define DDPF_FOURCC                 0x00000004
#define DDPF_INDEXED                0x00000020
#define DDPF_RGB                    0x00000040
#define DDPF_RGBA                   0x00000041  // DDPF_RGB | DDPF_ALPHAPIXELS
#define DDPF_YUV                    0x00000200
#define DDPF_LUMINANCE              0x00020000

//  DDS_HEADER.dwSurfaceFlags
#define DDSCAPS_COMPLEX             0x00000008
#define DDSCAPS_TEXTURE             0x00001000
#define DDSCAPS_MIPMAP              0x00400000

//  DDS_HEADER.dwSurfaceFlags2
#define DDSCAPS2_CUBEMAP            0x00000200
#define DDSCAPS2_CUBEMAP_POSITIVEX  0x00000400
#define DDSCAPS2_CUBEMAP_NEGATIVEX  0x00000800
#define DDSCAPS2_CUBEMAP_POSITIVEY  0x00001000
#define DDSCAPS2_CUBEMAP_NEGATIVEY  0x00002000
#define DDSCAPS2_CUBEMAP_POSITIVEZ  0x00004000
#define DDSCAPS2_CUBEMAP_NEGATIVEZ  0x00008000
#define DDSCAPS2_VOLUME             0x00200000

namespace {

enum D3DFORMAT
{
    D3DFMT_UNKNOWN              =  0,

    D3DFMT_R8G8B8               = 20,
    D3DFMT_A8R8G8B8             = 21,
    D3DFMT_X8R8G8B8             = 22,
    D3DFMT_R5G6B5               = 23,
    D3DFMT_X1R5G5B5             = 24,
    D3DFMT_A1R5G5B5             = 25,
    D3DFMT_A4R4G4B4             = 26,
    D3DFMT_R3G3B2               = 27,
    D3DFMT_A8                   = 28,
    D3DFMT_A8R3G3B2             = 29,
    D3DFMT_X4R4G4B4             = 30,
    D3DFMT_A2B10G10R10          = 31,
    D3DFMT_A8B8G8R8             = 32,
    D3DFMT_X8B8G8R8             = 33,
    D3DFMT_G16R16               = 34,
    D3DFMT_A2R10G10B10          = 35,
    D3DFMT_A16B16G16R16         = 36,

    D3DFMT_A8P8                 = 40,
    D3DFMT_P8                   = 41,

    D3DFMT_L8                   = 50,
    D3DFMT_A8L8                 = 51,
    D3DFMT_A4L4                 = 52,

    D3DFMT_V8U8                 = 60,
    D3DFMT_L6V5U5               = 61,
    D3DFMT_X8L8V8U8             = 62,
    D3DFMT_Q8W8V8U8             = 63,
    D3DFMT_V16U16               = 64,
    D3DFMT_A2W10V10U10          = 67,

    D3DFMT_UYVY                 = SCM_MAKEFOURCC('U', 'Y', 'V', 'Y'),
    D3DFMT_R8G8_B8G8            = SCM_MAKEFOURCC('R', 'G', 'B', 'G'),
    D3DFMT_YUY2                 = SCM_MAKEFOURCC('Y', 'U', 'Y', '2'),
    D3DFMT_G8R8_G8B8            = SCM_MAKEFOURCC('G', 'R', 'G', 'B'),
    D3DFMT_DXT1                 = SCM_MAKEFOURCC('D', 'X', 'T', '1'),
    D3DFMT_DXT2                 = SCM_MAKEFOURCC('D', 'X', 'T', '2'),
    D3DFMT_DXT3                 = SCM_MAKEFOURCC('D', 'X', 'T', '3'),
    D3DFMT_DXT4                 = SCM_MAKEFOURCC('D', 'X', 'T', '4'),
    D3DFMT_DXT5                 = SCM_MAKEFOURCC('D', 'X', 'T', '5'),

    D3DFMT_D16_LOCKABLE         = 70,
    D3DFMT_D32                  = 71,
    D3DFMT_D15S1                = 73,
    D3DFMT_D24S8                = 75,
    D3DFMT_D24X8                = 77,
    D3DFMT_D24X4S4              = 79,
    D3DFMT_D16                  = 80,

    D3DFMT_D32F_LOCKABLE        = 82,
    D3DFMT_D24FS8               = 83,

    D3DFMT_L16                  = 81,

    D3DFMT_VERTEXDATA           =100,
    D3DFMT_INDEX16              =101,
    D3DFMT_INDEX32              =102,

    D3DFMT_Q16W16V16U16         =110,

    D3DFMT_MULTI2_ARGB8         = SCM_MAKEFOURCC('M','E','T','1'),

    // Floating point surface formats

    // s10e5 formats (16-bits per channel)
    D3DFMT_R16F                 = 111,
    D3DFMT_G16R16F              = 112,
    D3DFMT_A16B16G16R16F        = 113,

    // IEEE s23e8 formats (32-bits per channel)
    D3DFMT_R32F                 = 114,
    D3DFMT_G32R32F              = 115,
    D3DFMT_A32B32G32R32F        = 116,

    D3DFMT_CxV8U8               = 117,

    D3DFMT_FORCE_DWORD          = 0x7fffffff
};


enum DXGI_FORMAT 
{
    DXGI_FORMAT_UNKNOWN                      = 0,
    DXGI_FORMAT_R32G32B32A32_TYPELESS        = 1,
    DXGI_FORMAT_R32G32B32A32_FLOAT           = 2,
    DXGI_FORMAT_R32G32B32A32_UINT            = 3,
    DXGI_FORMAT_R32G32B32A32_SINT            = 4,
    DXGI_FORMAT_R32G32B32_TYPELESS           = 5,
    DXGI_FORMAT_R32G32B32_FLOAT              = 6,
    DXGI_FORMAT_R32G32B32_UINT               = 7,
    DXGI_FORMAT_R32G32B32_SINT               = 8,
    DXGI_FORMAT_R16G16B16A16_TYPELESS        = 9,
    DXGI_FORMAT_R16G16B16A16_FLOAT           = 10,
    DXGI_FORMAT_R16G16B16A16_UNORM           = 11,
    DXGI_FORMAT_R16G16B16A16_UINT            = 12,
    DXGI_FORMAT_R16G16B16A16_SNORM           = 13,
    DXGI_FORMAT_R16G16B16A16_SINT            = 14,
    DXGI_FORMAT_R32G32_TYPELESS              = 15,
    DXGI_FORMAT_R32G32_FLOAT                 = 16,
    DXGI_FORMAT_R32G32_UINT                  = 17,
    DXGI_FORMAT_R32G32_SINT                  = 18,
    DXGI_FORMAT_R32G8X24_TYPELESS            = 19,
    DXGI_FORMAT_D32_FLOAT_S8X24_UINT         = 20,
    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS     = 21,
    DXGI_FORMAT_X32_TYPELESS_G8X24_UINT      = 22,
    DXGI_FORMAT_R10G10B10A2_TYPELESS         = 23,
    DXGI_FORMAT_R10G10B10A2_UNORM            = 24,
    DXGI_FORMAT_R10G10B10A2_UINT             = 25,
    DXGI_FORMAT_R11G11B10_FLOAT              = 26,
    DXGI_FORMAT_R8G8B8A8_TYPELESS            = 27,
    DXGI_FORMAT_R8G8B8A8_UNORM               = 28,
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB          = 29,
    DXGI_FORMAT_R8G8B8A8_UINT                = 30,
    DXGI_FORMAT_R8G8B8A8_SNORM               = 31,
    DXGI_FORMAT_R8G8B8A8_SINT                = 32,
    DXGI_FORMAT_R16G16_TYPELESS              = 33,
    DXGI_FORMAT_R16G16_FLOAT                 = 34,
    DXGI_FORMAT_R16G16_UNORM                 = 35,
    DXGI_FORMAT_R16G16_UINT                  = 36,
    DXGI_FORMAT_R16G16_SNORM                 = 37,
    DXGI_FORMAT_R16G16_SINT                  = 38,
    DXGI_FORMAT_R32_TYPELESS                 = 39,
    DXGI_FORMAT_D32_FLOAT                    = 40,
    DXGI_FORMAT_R32_FLOAT                    = 41,
    DXGI_FORMAT_R32_UINT                     = 42,
    DXGI_FORMAT_R32_SINT                     = 43,
    DXGI_FORMAT_R24G8_TYPELESS               = 44,
    DXGI_FORMAT_D24_UNORM_S8_UINT            = 45,
    DXGI_FORMAT_R24_UNORM_X8_TYPELESS        = 46,
    DXGI_FORMAT_X24_TYPELESS_G8_UINT         = 47,
    DXGI_FORMAT_R8G8_TYPELESS                = 48,
    DXGI_FORMAT_R8G8_UNORM                   = 49,
    DXGI_FORMAT_R8G8_UINT                    = 50,
    DXGI_FORMAT_R8G8_SNORM                   = 51,
    DXGI_FORMAT_R8G8_SINT                    = 52,
    DXGI_FORMAT_R16_TYPELESS                 = 53,
    DXGI_FORMAT_R16_FLOAT                    = 54,
    DXGI_FORMAT_D16_UNORM                    = 55,
    DXGI_FORMAT_R16_UNORM                    = 56,
    DXGI_FORMAT_R16_UINT                     = 57,
    DXGI_FORMAT_R16_SNORM                    = 58,
    DXGI_FORMAT_R16_SINT                     = 59,
    DXGI_FORMAT_R8_TYPELESS                  = 60,
    DXGI_FORMAT_R8_UNORM                     = 61,
    DXGI_FORMAT_R8_UINT                      = 62,
    DXGI_FORMAT_R8_SNORM                     = 63,
    DXGI_FORMAT_R8_SINT                      = 64,
    DXGI_FORMAT_A8_UNORM                     = 65,
    DXGI_FORMAT_R1_UNORM                     = 66,
    DXGI_FORMAT_R9G9B9E5_SHAREDEXP           = 67,
    DXGI_FORMAT_R8G8_B8G8_UNORM              = 68,
    DXGI_FORMAT_G8R8_G8B8_UNORM              = 69,
    DXGI_FORMAT_BC1_TYPELESS                 = 70,
    DXGI_FORMAT_BC1_UNORM                    = 71,
    DXGI_FORMAT_BC1_UNORM_SRGB               = 72,
    DXGI_FORMAT_BC2_TYPELESS                 = 73,
    DXGI_FORMAT_BC2_UNORM                    = 74,
    DXGI_FORMAT_BC2_UNORM_SRGB               = 75,
    DXGI_FORMAT_BC3_TYPELESS                 = 76,
    DXGI_FORMAT_BC3_UNORM                    = 77,
    DXGI_FORMAT_BC3_UNORM_SRGB               = 78,
    DXGI_FORMAT_BC4_TYPELESS                 = 79,
    DXGI_FORMAT_BC4_UNORM                    = 80,
    DXGI_FORMAT_BC4_SNORM                    = 81,
    DXGI_FORMAT_BC5_TYPELESS                 = 82,
    DXGI_FORMAT_BC5_UNORM                    = 83,
    DXGI_FORMAT_BC5_SNORM                    = 84,
    DXGI_FORMAT_B5G6R5_UNORM                 = 85,
    DXGI_FORMAT_B5G5R5A1_UNORM               = 86,
    DXGI_FORMAT_B8G8R8A8_UNORM               = 87,
    DXGI_FORMAT_B8G8R8X8_UNORM               = 88,
    DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM   = 89,
    DXGI_FORMAT_B8G8R8A8_TYPELESS            = 90,
    DXGI_FORMAT_B8G8R8A8_UNORM_SRGB          = 91,
    DXGI_FORMAT_B8G8R8X8_TYPELESS            = 92,
    DXGI_FORMAT_B8G8R8X8_UNORM_SRGB          = 93,
    DXGI_FORMAT_BC6H_TYPELESS                = 94,
    DXGI_FORMAT_BC6H_UF16                    = 95,
    DXGI_FORMAT_BC6H_SF16                    = 96,
    DXGI_FORMAT_BC7_TYPELESS                 = 97,
    DXGI_FORMAT_BC7_UNORM                    = 98,
    DXGI_FORMAT_BC7_UNORM_SRGB               = 99,
    DXGI_FORMAT_FORCE_UINT                   = 0xffffffffUL
};

static const int DXGI_FORMAT_COUNT = 101;

enum D3D10_RESOURCE_DIMENSION 
{
    D3D10_RESOURCE_DIMENSION_UNKNOWN     = 0,
    D3D10_RESOURCE_DIMENSION_BUFFER      = 1,
    D3D10_RESOURCE_DIMENSION_TEXTURE1D   = 2,
    D3D10_RESOURCE_DIMENSION_TEXTURE2D   = 3,
    D3D10_RESOURCE_DIMENSION_TEXTURE3D   = 4 
};

enum D3D10_RESOURCE_MISC_FLAG 
{
    D3D10_RESOURCE_MISC_GENERATE_MIPS       = 0x00000001,
    D3D10_RESOURCE_MISC_SHARED              = 0x00000002,
    D3D10_RESOURCE_MISC_TEXTURECUBE         = 0x00000004,
    D3D10_RESOURCE_MISC_SHARED_KEYEDMUTEX   = 0x00000010,
    D3D10_RESOURCE_MISC_GDI_COMPATIBLE      = 0x00000020
};

struct DDS_PIXELFORMAT
{
    unsigned int dwSize;
    unsigned int dwFlags;
    unsigned int dwFourCC;
    unsigned int dwRGBBitCount;
    unsigned int dwRBitMask;
    unsigned int dwGBitMask;
    unsigned int dwBBitMask;
    unsigned int dwABitMask;
}; // struct DDS_PIXELFORMAT

struct DDS_HEADER
{
    unsigned int    dwSize;
    unsigned int    dwHeaderFlags;
    unsigned int    dwHeight;
    unsigned int    dwWidth;
    unsigned int    dwPitchOrLinearSize;
    unsigned int    dwDepth; // only if DDS_HEADER_FLAGS_VOLUME is set in dwHeaderFlags
    unsigned int    dwMipMapCount;
    unsigned int    dwReserved1[11];
    DDS_PIXELFORMAT ddspf;
    unsigned int    dwSurfaceFlags;
    unsigned int    dwSurfaceFlags2;
    unsigned int    dwReserved2[3];
}; // struct DDS_HEADER

struct DDS_HEADER_DXT10
{
    DXGI_FORMAT dxgiFormat;
    D3D10_RESOURCE_DIMENSION resourceDimension;
    unsigned int miscFlag;
    unsigned int arraySize;
    unsigned int reserved;
}; // struct DDS_HEADER_DXT10;

class dds_file
{
public:
    dds_file(const std::string&   in_file_name)
    : _image_data_offset(0)
    , _image_data_size(0)
    {
        using namespace scm;
        using namespace scm::gl;
        using namespace scm::io;

        _file.reset(new io::file());

        // TODO check if file exists...

        if (!_file->open(in_file_name, std::ios_base::in, false)) {
            glerr() << log::error
                    << "dds_file::dds_file(): error opening file: "
                    << in_file_name << log::end;
            cleanup();
            return;
        }

        scm::size_t       dds_header_off        = sizeof(unsigned int);
        scm::size_t       dds_header_dxt10_off  = dds_header_off + sizeof(DDS_HEADER);

        // check  magic number ("DDS ")
        unsigned int magic_number = 0;
        if (_file->read(&magic_number, 0, sizeof(unsigned int)) != sizeof(unsigned int)) {
            glerr() << log::error
                    << "dds_file::dds_file(): error reading from file: " << in_file_name 
                    << " (number of bytes attempted to read: " << sizeof(unsigned int) << ", at position : " << 0 << ")" << log::end;
            cleanup();
            return;
        }

        if (magic_number != DDS_MAGIC) {
            glerr() << log::error
                    << "dds_file::dds_file(): error file not starting with magic \"dds \" number: "
                    << in_file_name << log::end;
            cleanup();
            return;
        }

        _dds_header.reset(new DDS_HEADER);
        if (_file->read(_dds_header.get(), dds_header_off, sizeof(DDS_HEADER)) != sizeof(DDS_HEADER)) {
            glerr() << log::error
                    << "dds_file::dds_file(): error reading from file: " << in_file_name 
                    << " (number of bytes attempted to read: " << sizeof(DDS_HEADER) << ", at position : " << dds_header_off << ")" << log::end;
            cleanup();
            return;
        }
        if (   _dds_header->dwSize       != sizeof(DDS_HEADER)
            || _dds_header->ddspf.dwSize != sizeof(DDS_PIXELFORMAT)) {
            glerr() << log::error
                    << "dds_file::dds_file(): error validating header size information: "
                    << in_file_name << log::end;
            cleanup();
            return;
        }

        // check for DX10 extension
        //if (   (_dds_header->ddspf.dwFlags & DDPF_FOURCC)
        //    && (_dds_header->ddspf.dwFourCC == SCM_MAKEFOURCC('D', 'X' ,'1' ,'0'))) {
        if (_dds_header->ddspf.dwFourCC == SCM_MAKEFOURCC('D', 'X' ,'1' ,'0')) {
            // Must be long enough for both headers and magic value
            _dds_header_dxt10.reset(new DDS_HEADER_DXT10);

            if (_file->read(_dds_header_dxt10.get(), dds_header_dxt10_off, sizeof(DDS_HEADER_DXT10)) != sizeof(DDS_HEADER_DXT10)) {
                glerr() << log::error
                        << "dds_file::dds_file(): error reading from file: " << in_file_name 
                        << " (number of bytes attempted to read: " << sizeof(DDS_HEADER_DXT10) << ", at position : " << dds_header_dxt10_off << ")" << log::end;
                cleanup();
                return;
            }
        }

        if (_dds_header_dxt10) {
            _image_data_offset = dds_header_dxt10_off + sizeof(DDS_HEADER_DXT10);
        }
        else {
            _image_data_offset = dds_header_off + sizeof(DDS_HEADER);
        }

        _image_data_size = _file->size() - _image_data_offset;
    }

    const scm::shared_ptr<scm::io::file>            file() const              { assert(_file);                  return _file; }
    const scm::shared_ptr<const DDS_HEADER>         dds_header() const        { assert(_dds_header);            return _dds_header; }
    const scm::shared_ptr<const DDS_HEADER_DXT10>   dds_header_dxt10() const  {                                 return _dds_header_dxt10; }
    scm::size_t                                     image_data_offset() const { assert(_image_data_offset > 0); return _image_data_offset; }
    scm::size_t                                     image_data_size() const   {                                 return _image_data_size; }

                                                    operator bool() const     { return _file.get() != 0; }
    bool                                            operator! () const        { return _file.get() == 0; }

private:
    scm::shared_ptr<scm::io::file>      _file;
    scm::shared_ptr<DDS_HEADER>         _dds_header;
    scm::shared_ptr<DDS_HEADER_DXT10>   _dds_header_dxt10;
    scm::size_t                         _image_data_offset;
    scm::size_t                         _image_data_size;

    void cleanup() {
        _file.reset();
        _dds_header.reset();
        _dds_header_dxt10.reset();
        _image_data_offset = 0;
        _image_data_size   = 0;
    }
};

#define SCM_ISBITMASK(r, g, b, a) \
    (   dds.dds_header()->ddspf.dwRBitMask == r \
     && dds.dds_header()->ddspf.dwGBitMask == g \
     && dds.dds_header()->ddspf.dwBBitMask == b \
     && dds.dds_header()->ddspf.dwABitMask == a)

scm::gl::data_format
match_format(const dds_file& dds)
{
    using namespace scm;
    using namespace scm::gl;


    if (dds.dds_header_dxt10()) {
        switch (dds.dds_header_dxt10()->dxgiFormat) {
            case DXGI_FORMAT_UNKNOWN                      : return FORMAT_NULL;            // = 0,
            case DXGI_FORMAT_R32G32B32A32_TYPELESS        : return FORMAT_RGBA_32UI;       // = 1,
            case DXGI_FORMAT_R32G32B32A32_FLOAT           : return FORMAT_RGBA_32F;        // = 2,
            case DXGI_FORMAT_R32G32B32A32_UINT            : return FORMAT_RGBA_32UI;       // = 3,
            case DXGI_FORMAT_R32G32B32A32_SINT            : return FORMAT_RGBA_32I;        // = 4,
            case DXGI_FORMAT_R32G32B32_TYPELESS           : return FORMAT_RGB_32UI;        // = 5,
            case DXGI_FORMAT_R32G32B32_FLOAT              : return FORMAT_RGB_32F;         // = 6,
            case DXGI_FORMAT_R32G32B32_UINT               : return FORMAT_RGB_32UI;        // = 7,
            case DXGI_FORMAT_R32G32B32_SINT               : return FORMAT_RGB_32I;         // = 8,
            case DXGI_FORMAT_R16G16B16A16_TYPELESS        : return FORMAT_RGBA_16UI;       // = 9,
            case DXGI_FORMAT_R16G16B16A16_FLOAT           : return FORMAT_RGBA_16F;        // = 10,
            case DXGI_FORMAT_R16G16B16A16_UNORM           : return FORMAT_RGBA_16;         // = 11,
            case DXGI_FORMAT_R16G16B16A16_UINT            : return FORMAT_RGBA_16UI;       // = 12,
            case DXGI_FORMAT_R16G16B16A16_SNORM           : return FORMAT_RGBA_16S;        // = 13,
            case DXGI_FORMAT_R16G16B16A16_SINT            : return FORMAT_RGBA_16I;        // = 14,
            case DXGI_FORMAT_R32G32_TYPELESS              : return FORMAT_RG_32UI;         // = 15,
            case DXGI_FORMAT_R32G32_FLOAT                 : return FORMAT_RG_32F;          // = 16,
            case DXGI_FORMAT_R32G32_UINT                  : return FORMAT_RG_32UI;         // = 17,
            case DXGI_FORMAT_R32G32_SINT                  : return FORMAT_RG_32I;          // = 18,
            case DXGI_FORMAT_R32G8X24_TYPELESS            : return FORMAT_NULL;            // = 19,
            case DXGI_FORMAT_D32_FLOAT_S8X24_UINT         : return FORMAT_D32F_S8;         // = 20,
            case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS     : return FORMAT_NULL;            // = 21,
            case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT      : return FORMAT_NULL;            // = 22,
            case DXGI_FORMAT_R10G10B10A2_TYPELESS         : return FORMAT_NULL;            // = 23,
            case DXGI_FORMAT_R10G10B10A2_UNORM            : return FORMAT_NULL;            // = 24,
            case DXGI_FORMAT_R10G10B10A2_UINT             : return FORMAT_NULL;            // = 25,
            case DXGI_FORMAT_R11G11B10_FLOAT              : return FORMAT_R11B11G10F;      // = 26,
            case DXGI_FORMAT_R8G8B8A8_TYPELESS            : return FORMAT_RGBA_8UI;        // = 27,
            case DXGI_FORMAT_R8G8B8A8_UNORM               : return FORMAT_RGBA_8;          // = 28,
            case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB          : return FORMAT_SRGBA_8;         // = 29,
            case DXGI_FORMAT_R8G8B8A8_UINT                : return FORMAT_RGBA_8UI;        // = 30,
            case DXGI_FORMAT_R8G8B8A8_SNORM               : return FORMAT_RGBA_8S;         // = 31,
            case DXGI_FORMAT_R8G8B8A8_SINT                : return FORMAT_RGBA_8I;         // = 32,
            case DXGI_FORMAT_R16G16_TYPELESS              : return FORMAT_RG_16UI;         // = 33,
            case DXGI_FORMAT_R16G16_FLOAT                 : return FORMAT_RG_16F;          // = 34,
            case DXGI_FORMAT_R16G16_UNORM                 : return FORMAT_RG_16;           // = 35,
            case DXGI_FORMAT_R16G16_UINT                  : return FORMAT_RG_16UI;         // = 36,
            case DXGI_FORMAT_R16G16_SNORM                 : return FORMAT_RG_16S;          // = 37,
            case DXGI_FORMAT_R16G16_SINT                  : return FORMAT_RG_16I;          // = 38,
            case DXGI_FORMAT_R32_TYPELESS                 : return FORMAT_R_32UI;          // = 39,
            case DXGI_FORMAT_D32_FLOAT                    : return FORMAT_D32;             // = 40,
            case DXGI_FORMAT_R32_FLOAT                    : return FORMAT_R_32F;           // = 41,
            case DXGI_FORMAT_R32_UINT                     : return FORMAT_R_32UI;          // = 42,
            case DXGI_FORMAT_R32_SINT                     : return FORMAT_R_32I;           // = 43,
            case DXGI_FORMAT_R24G8_TYPELESS               : return FORMAT_NULL;            // = 44,
            case DXGI_FORMAT_D24_UNORM_S8_UINT            : return FORMAT_NULL;            // = 45,
            case DXGI_FORMAT_R24_UNORM_X8_TYPELESS        : return FORMAT_NULL;            // = 46,
            case DXGI_FORMAT_X24_TYPELESS_G8_UINT         : return FORMAT_NULL;            // = 47,
            case DXGI_FORMAT_R8G8_TYPELESS                : return FORMAT_RG_8UI;          // = 48,
            case DXGI_FORMAT_R8G8_UNORM                   : return FORMAT_RG_8;            // = 49,
            case DXGI_FORMAT_R8G8_UINT                    : return FORMAT_RG_8UI;          // = 50,
            case DXGI_FORMAT_R8G8_SNORM                   : return FORMAT_RG_8S;           // = 51,
            case DXGI_FORMAT_R8G8_SINT                    : return FORMAT_RG_8I;           // = 52,
            case DXGI_FORMAT_R16_TYPELESS                 : return FORMAT_R_16UI;          // = 53,
            case DXGI_FORMAT_R16_FLOAT                    : return FORMAT_R_16F;           // = 54,
            case DXGI_FORMAT_D16_UNORM                    : return FORMAT_D16;             // = 55,
            case DXGI_FORMAT_R16_UNORM                    : return FORMAT_R_16;            // = 56,
            case DXGI_FORMAT_R16_UINT                     : return FORMAT_R_16UI;          // = 57,
            case DXGI_FORMAT_R16_SNORM                    : return FORMAT_R_16S;           // = 58,
            case DXGI_FORMAT_R16_SINT                     : return FORMAT_R_16I;           // = 59,
            case DXGI_FORMAT_R8_TYPELESS                  : return FORMAT_R_8UI;           // = 60,
            case DXGI_FORMAT_R8_UNORM                     : return FORMAT_R_8;             // = 61,
            case DXGI_FORMAT_R8_UINT                      : return FORMAT_R_8UI;           // = 62,
            case DXGI_FORMAT_R8_SNORM                     : return FORMAT_R_8S;            // = 63,
            case DXGI_FORMAT_R8_SINT                      : return FORMAT_R_8I;            // = 64,
            case DXGI_FORMAT_A8_UNORM                     : return FORMAT_R_8;             // = 65,
            case DXGI_FORMAT_R1_UNORM                     : return FORMAT_NULL;            // = 66,
            case DXGI_FORMAT_R9G9B9E5_SHAREDEXP           : return FORMAT_RGB9_E5;         // = 67,
            case DXGI_FORMAT_R8G8_B8G8_UNORM              : return FORMAT_NULL;            // = 68,
            case DXGI_FORMAT_G8R8_G8B8_UNORM              : return FORMAT_NULL;            // = 69,
            case DXGI_FORMAT_BC1_TYPELESS                 : return FORMAT_BC1_RGBA;        // = 70,
            case DXGI_FORMAT_BC1_UNORM                    : return FORMAT_BC1_RGBA;        // = 71,
            case DXGI_FORMAT_BC1_UNORM_SRGB               : return FORMAT_BC1_SRGBA;       // = 72,
            case DXGI_FORMAT_BC2_TYPELESS                 : return FORMAT_BC2_RGBA;        // = 73,
            case DXGI_FORMAT_BC2_UNORM                    : return FORMAT_BC2_RGBA;        // = 74,
            case DXGI_FORMAT_BC2_UNORM_SRGB               : return FORMAT_BC2_SRGBA;       // = 75,
            case DXGI_FORMAT_BC3_TYPELESS                 : return FORMAT_BC3_RGBA;        // = 76,
            case DXGI_FORMAT_BC3_UNORM                    : return FORMAT_BC3_RGBA;        // = 77,
            case DXGI_FORMAT_BC3_UNORM_SRGB               : return FORMAT_BC3_SRGBA;       // = 78,
            case DXGI_FORMAT_BC4_TYPELESS                 : return FORMAT_BC4_R;           // = 79,
            case DXGI_FORMAT_BC4_UNORM                    : return FORMAT_BC4_R;           // = 80,
            case DXGI_FORMAT_BC4_SNORM                    : return FORMAT_BC4_R_S;         // = 81,
            case DXGI_FORMAT_BC5_TYPELESS                 : return FORMAT_BC5_RG;          // = 82,
            case DXGI_FORMAT_BC5_UNORM                    : return FORMAT_BC5_RG;          // = 83,
            case DXGI_FORMAT_BC5_SNORM                    : return FORMAT_BC5_RG_S;        // = 84,
            case DXGI_FORMAT_B5G6R5_UNORM                 : return FORMAT_NULL;            // = 85,
            case DXGI_FORMAT_B5G5R5A1_UNORM               : return FORMAT_NULL;            // = 86,
            case DXGI_FORMAT_B8G8R8A8_UNORM               : return FORMAT_BGRA_8;          // = 87,
            case DXGI_FORMAT_B8G8R8X8_UNORM               : return FORMAT_BGRA_8;          // = 88,
            case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM   : return FORMAT_NULL;            // = 89,
            case DXGI_FORMAT_B8G8R8A8_TYPELESS            : return FORMAT_BGRA_8;          // = 90,
            case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB          : return FORMAT_BGRA_8;          // = 91,
            case DXGI_FORMAT_B8G8R8X8_TYPELESS            : return FORMAT_BGRA_8;          // = 92,
            case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB          : return FORMAT_BGRA_8;          // = 93,
            case DXGI_FORMAT_BC6H_TYPELESS                : return FORMAT_BC6H_RGB_UF;     // = 94,
            case DXGI_FORMAT_BC6H_UF16                    : return FORMAT_BC6H_RGB_UF;     // = 95,
            case DXGI_FORMAT_BC6H_SF16                    : return FORMAT_BC6H_RGB_F;      // = 96,
            case DXGI_FORMAT_BC7_TYPELESS                 : return FORMAT_BC7_RGBA;        // = 97,
            case DXGI_FORMAT_BC7_UNORM                    : return FORMAT_BC7_RGBA;        // = 98,
            case DXGI_FORMAT_BC7_UNORM_SRGB               : return FORMAT_BC7_SRGBA;       // = 99,
            case DXGI_FORMAT_FORCE_UINT                   : return FORMAT_R_32UI;          // = 0xffffffffUL
            default                                       : return FORMAT_NULL;
        };
    }
    else if(dds.dds_header()->ddspf.dwFlags & DDPF_FOURCC) {
        switch (dds.dds_header()->ddspf.dwFourCC) {
            case D3DFMT_R8G8B8               : return FORMAT_BGR_8;
            case D3DFMT_A8R8G8B8             : return FORMAT_BGRA_8;
            case D3DFMT_X8R8G8B8             : return FORMAT_BGRA_8;
            case D3DFMT_A8                   : return FORMAT_R_8;
            case D3DFMT_A2B10G10R10          : return FORMAT_NULL;
            case D3DFMT_A8B8G8R8             : return FORMAT_RGBA_8;
            case D3DFMT_X8B8G8R8             : return FORMAT_RGBA_8;
            case D3DFMT_G16R16               : return FORMAT_RG_16;
            case D3DFMT_A2R10G10B10          : return FORMAT_NULL;
            case D3DFMT_A16B16G16R16         : return FORMAT_RGBA_16;
            case D3DFMT_L8                   : return FORMAT_R_8;
            case D3DFMT_A8L8                 : return FORMAT_RG_8;
            case D3DFMT_DXT1                 : return FORMAT_BC1_RGBA;
            case D3DFMT_DXT2                 : return FORMAT_BC2_RGBA; // pre-mult alpha
            case D3DFMT_DXT3                 : return FORMAT_BC2_RGBA;
            case D3DFMT_DXT4                 : return FORMAT_BC3_RGBA; // pre-mult alpha
            case D3DFMT_DXT5                 : return FORMAT_BC3_RGBA;
            case D3DFMT_L16                  : return FORMAT_R_16;
            case D3DFMT_R16F                 : return FORMAT_R_16F;
            case D3DFMT_G16R16F              : return FORMAT_RG_16F;
            case D3DFMT_A16B16G16R16F        : return FORMAT_RGBA_16F;
            case D3DFMT_R32F                 : return FORMAT_R_32F;
            case D3DFMT_G32R32F              : return FORMAT_RG_32F;
            case D3DFMT_A32B32G32R32F        : return FORMAT_RGBA_32F;
            default                          : return FORMAT_NULL;
        }
    }
    else if (   dds.dds_header()->ddspf.dwFlags & DDPF_RGB
             || dds.dds_header()->ddspf.dwFlags & DDPF_RGBA) {
        switch (dds.dds_header()->ddspf.dwRGBBitCount) {
            case 16:break;
            case 24:
                if(SCM_ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000))   return FORMAT_RGB_8;
                if(SCM_ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000))   return FORMAT_BGR_8;
                break;
            case 32:
                if(SCM_ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000))   return FORMAT_BGRA_8;
                if(SCM_ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000))   return FORMAT_BGRA_8;
                if(SCM_ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))   return FORMAT_RGBA_8;
                if(SCM_ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000))   return FORMAT_RGBA_8;

                // Note that many common DDS reader/writers swap the
                // the RED/BLUE masks for 10:10:10:2 formats. We assumme
                // below that the 'correct' header mask is being used
                //if(SCM_ISBITMASK(0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000))   return FORMAT_BGR10A2;
                //if(SCM_ISBITMASK(0x000003ff, 0x000ffc00, 0x3ff00000, 0xc0000000))   return FORMAT_RGB10A2;

                if(SCM_ISBITMASK(0x0000ffff, 0xffff0000, 0x00000000, 0x00000000))   return FORMAT_RG_16;
                if(SCM_ISBITMASK(0xffffffff, 0x00000000, 0x00000000, 0x00000000))   return FORMAT_R_32F;
                break;
            default: return FORMAT_NULL;
        }
    }
    else if (dds.dds_header()->ddspf.dwFlags & DDPF_LUMINANCE) {
        switch (dds.dds_header()->ddspf.dwRGBBitCount) {
            case 8:
                if(SCM_ISBITMASK(0x000000ff, 0x00000000, 0x00000000, 0x00000000))   return FORMAT_R_8;
                break;
            case 16:
                if(SCM_ISBITMASK(0x0000ffff, 0x00000000, 0x00000000, 0x00000000))   return FORMAT_R_16;
                break;
            default: return FORMAT_NULL;
        }
    }
    else if (   dds.dds_header()->ddspf.dwFlags & DDPF_ALPHA
             || dds.dds_header()->ddspf.dwFlags & DDPF_ALPHAPIXELS) {
        switch (dds.dds_header()->ddspf.dwRGBBitCount) {
            case 8:  return FORMAT_R_8; break;
            case 16: return FORMAT_R_16;break;
            default: return FORMAT_NULL;
        }
    }

    return FORMAT_NULL;
}

#undef SCM_ISBITMASK

unsigned
dds_fourcc(scm::gl::data_format f)
{
    using namespace scm::gl;
#if 1
    switch (f) {
        //case FORMAT_BGR_8       : return D3DFMT_R8G8B8;
        //case FORMAT_BGRA_8      : return D3DFMT_A8R8G8B8;
        //case FORMAT_R_8         : return D3DFMT_A8;
        //case FORMAT_RGBA_8      : return D3DFMT_A8B8G8R8;
        //case FORMAT_RG_16       : return D3DFMT_G16R16;
        //case FORMAT_RGBA_16     : return D3DFMT_A16B16G16R16;
        //case FORMAT_RG_8        : return D3DFMT_A8L8;
        case FORMAT_BC1_RGBA    : return D3DFMT_DXT1;
        case FORMAT_BC2_RGBA    : return D3DFMT_DXT3;
        case FORMAT_BC3_RGBA    : return D3DFMT_DXT5;
        //case FORMAT_R_16        : return D3DFMT_L16;
        case FORMAT_R_16F       : return D3DFMT_R16F;
        case FORMAT_RG_16F      : return D3DFMT_G16R16F;
        case FORMAT_RGBA_16F    : return D3DFMT_A16B16G16R16F;
        case FORMAT_R_32F       : return D3DFMT_R32F;
        case FORMAT_RG_32F      : return D3DFMT_G32R32F;
        case FORMAT_RGBA_32F    : return D3DFMT_A32B32G32R32F;
        default                 : return 0;
    }
#else
    return 0;
#endif
}

bool
dds_bitmask(scm::gl::data_format f, unsigned& rm, unsigned& gm, unsigned& bm, unsigned& am)
{
    using namespace scm::gl;

    if (dds_fourcc(f) != 0) {
        rm = gm = bm = am = 0u;
        return true;
    }

    switch (f) {
        case FORMAT_R_8:    rm = 0x000000ffu; gm = 0x00000000u; bm = 0x00000000u; am = 0x00000000u; break;
        case FORMAT_R_16:   rm = 0x0000ffffu; gm = 0x00000000u; bm = 0x00000000u; am = 0x00000000u; break;
        case FORMAT_R_32F:  rm = 0xffffffffu; gm = 0x00000000u; bm = 0x00000000u; am = 0x00000000u; break;

        case FORMAT_RGB_8:  rm = 0x000000ffu; gm = 0x0000ff00u; bm = 0x00ff0000u; am = 0x00000000u; break;
        case FORMAT_BGR_8:  rm = 0x00ff0000u; gm = 0x0000ff00u; bm = 0x000000ffu; am = 0x00000000u; break;

        case FORMAT_RGBA_8: rm = 0x000000ffu; gm = 0x0000ff00u; bm = 0x00ff0000u; am = 0xff000000u; break;
        case FORMAT_BGRA_8: rm = 0x00ff0000u; gm = 0x0000ff00u; bm = 0x000000ffu; am = 0xff000000u; break;

        case FORMAT_RG_16:  rm = 0x0000ffffu; gm = 0xffff0000u; bm = 0x00000000u; am = 0x00000000u; break;

        default:            return false;
    };

    return true;
}

unsigned
dds_flags(scm::gl::data_format f)
{
    using namespace scm::gl;

    unsigned r = 0;
    if (dds_fourcc(f) != 0) {
        r |= DDPF_FOURCC;
    }
    else {
        switch (f) {
            case FORMAT_RG_16:
            case FORMAT_R_32F:
            case FORMAT_RGB_8:
            case FORMAT_BGR_8:
                r |= DDPF_RGB;
                break;
            case FORMAT_RGBA_8:
            case FORMAT_BGRA_8:
                r |= DDPF_RGBA;
                break;
            case FORMAT_R_8:
            case FORMAT_R_16:
                r |= DDPF_LUMINANCE;
            break;
        };
    }
    return r;
}

scm::math::vec3ui
retrieve_dimensions(const dds_file& dds)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    vec3ui s = vec3ui(0);

    s.x = dds.dds_header()->dwWidth;
    s.y = dds.dds_header()->dwHeight;
    s.z = dds.dds_header()->dwDepth;

    if (s.z == 0) {
        s.z = 1;
    }

    return s;
}

unsigned
retrieve_mipmap_count(const dds_file& dds)
{
    if (  !(dds.dds_header()->dwSurfaceFlags & DDSCAPS_MIPMAP)
        || (dds.dds_header()->dwMipMapCount == 0)) {
        return 1;
    }
    else {
        return dds.dds_header()->dwMipMapCount;
    }
}

unsigned
retrieve_layer_count(const dds_file& dds)
{
    if (dds.dds_header_dxt10()) {
        if (dds.dds_header()->dwSurfaceFlags2 & DDSCAPS2_VOLUME) {
            return 1;
        }
        else {
            return dds.dds_header_dxt10()->arraySize;
        }
    }
    else if (dds.dds_header()->dwSurfaceFlags2 & DDSCAPS2_CUBEMAP) {
        // 
        return 6;
    }
    else {
        return 1;
    }
}

scm::size_t
mip_level_size(const scm::math::vec3ui& lsize, scm::gl::data_format fmt)
{
    using namespace scm::math;

    if (is_compressed_format(fmt)) {
        scm::size_t w = (lsize.x + 3) / 4;
        scm::size_t h = (lsize.y + 3) / 4;
        scm::size_t d =  lsize.z;
        return w * h * d * scm::gl::compressed_block_size(fmt);
    }
    else {
        scm::size_t w = lsize.x;
        scm::size_t h = lsize.y;
        scm::size_t d = lsize.z;
        return w * h * d * scm::gl::size_of_format(fmt);
    }
}

} // namespace

namespace scm {
namespace gl {

texture_image_data_ptr
texture_loader_dds::load_image_data(const std::string& in_image_path) const
{
    using namespace scm::math;

    ::dds_file   raw_dds = ::dds_file(in_image_path);

    if (!raw_dds) {
        glerr() << log::error
                << "texture_loader_dds::load_image_data(): error reading dds image data." << log::end;
        return texture_image_data_ptr();
    }

    if (!(raw_dds.dds_header()->dwSurfaceFlags & DDSCAPS_TEXTURE)) {
        glerr() << log::error
                << "texture_loader_dds::load_image_data(): error DDSCAPS_TEXTURE not present." << log::end;
        return texture_image_data_ptr();
    }

    data_format img_format = match_format(raw_dds);
    if (img_format == FORMAT_NULL) {
        glerr() << log::error
                << "texture_loader_dds::load_image_data(): error matching dds data format to scm::gl::data_format." << log::end;
        return texture_image_data_ptr();
    }

    vec3ui   img_size           = retrieve_dimensions(raw_dds);
    unsigned img_mip_count      = retrieve_mipmap_count(raw_dds);
    unsigned img_layer_count    = retrieve_layer_count(raw_dds);

    texture_image_data::level_vector    img_lev_data;

    { // init image memory
        vec3ui lsize = img_size;
        for (unsigned l = 0; l < img_mip_count; ++l) {
            shared_array<uint8> ldata;
            scm::size_t         ldata_size =  mip_level_size(lsize, img_format) * img_layer_count;

            try {
                ldata.reset(new uint8[ldata_size]);
            }
            catch (const std::bad_alloc& e) {
                glerr() << log::error
                        << "texture_loader_dds::load_image_data(): error allocating image memory "
                        << "(level: " << l << ", size: " << lsize << ", layers: " << img_layer_count
                        << ", format: " << format_string(img_format) << ", ldata_size: " << ldata_size << "), "
                        << e.what() << log::end;
                return texture_image_data_ptr();
            }

            img_lev_data.push_back(texture_image_data::level(lsize, ldata));

            lsize.x = max(1u, lsize.x / 2);
            lsize.y = max(1u, lsize.y / 2);
            lsize.z = max(1u, lsize.z / 2);
        }
    }

    { // read image data
        io::file::offset_type roff = raw_dds.image_data_offset();
        for (unsigned a = 0; a < img_layer_count; ++a) {
            for (unsigned l = 0; l < img_mip_count; ++l) {
                const shared_array<uint8> ldata         = img_lev_data[l].data();
                const scm::size_t         lmip_img_size = mip_level_size(img_lev_data[l].size(), img_format);

                if ((roff + lmip_img_size - raw_dds.image_data_offset()) > raw_dds.image_data_size()) {
                    glerr() << log::error
                            << "texture_loader_dds::load_image_data(): error trying to read past file size: " << in_image_path 
                            << " (number of bytes attempted to read: " << lmip_img_size << ", at position : " << roff << ")" << log::end;
                    return texture_image_data_ptr();
                }

                if (raw_dds.file()->read(ldata.get() + lmip_img_size * a, roff, lmip_img_size) != lmip_img_size) {
                    glerr() << log::error
                            << "texture_loader_dds::load_image_data(): error reading from file: " << in_image_path 
                            << " (number of bytes attempted to read: " << lmip_img_size << ", at position : " << roff << ")" << log::end;
                    return texture_image_data_ptr();
                }

                roff += lmip_img_size;
            }
        }
    }

    texture_image_data_ptr ret_img(new texture_image_data(texture_image_data::ORIGIN_UPPER_LEFT, img_format, img_layer_count, img_lev_data));
    
    // dds files use upper-left origin, we flip!
    ret_img->flip_vertical();

    return ret_img;
}

bool
texture_loader_dds::save_image_data_dx9(const std::string&           in_image_path,
                                        const texture_image_data_ptr in_img_data) const
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::io;
    using namespace scm::math;

    if (in_img_data->origin() == texture_image_data::ORIGIN_LOWER_LEFT) {
        if (!in_img_data->flip_vertical()) {
            glerr() << log::error
                    << "texture_loader_dds::save_image_data_dx9(): error flipping image data before save operation." << log::end;
            return false;
        }
    }


    scoped_ptr<io::file> out_file(new io::file());

    // TODO check if file exists...

    if (!out_file->open(in_image_path, std::ios_base::out, false)) {
        glerr() << log::error
                << "texture_loader_dds::save_image_data_dx9(): error opening output file: "
                << in_image_path << log::end;
        return false;
    }

    scm::size_t       dds_header_off = sizeof(unsigned int);
    scm::size_t       dds_data_off   = dds_header_off + sizeof(DDS_HEADER);

    // check  magic number ("DDS ")
    unsigned int magic_number = DDS_MAGIC;
    if (out_file->write(&magic_number, 0, sizeof(unsigned int)) != sizeof(unsigned int)) {
        glerr() << log::error
                << "texture_loader_dds::save_image_data_dx9(): error writing to output file: " << in_image_path 
                << " (number of bytes attempted to write: " << sizeof(unsigned int) << ", at position : " << 0 << ")" << log::end;
        return false;
    }

    scoped_ptr<DDS_HEADER> dds9_header(new DDS_HEADER);
    memset(dds9_header.get(), 0, sizeof(DDS_HEADER));

    // setup dds header
    dds9_header->dwSize              = sizeof(DDS_HEADER);
    dds9_header->dwHeaderFlags       =   DDSD_CAPS
                                       | DDSD_WIDTH
                                       | DDSD_HEIGHT
                                       | (in_img_data->mip_level(0).size().z > 1 ? DDSD_DEPTH : 0)
                                       | DDSD_PIXELFORMAT
                                       | (in_img_data->mip_level_count() > 1 ? DDSD_MIPMAPCOUNT : 0)
                                       | (is_compressed_format(in_img_data->format()) ? DDSD_LINEARSIZE : DDSD_PITCH);
    dds9_header->dwHeight            = in_img_data->mip_level(0).size().y;
    dds9_header->dwWidth             = in_img_data->mip_level(0).size().x;
    dds9_header->dwPitchOrLinearSize = (is_compressed_format(in_img_data->format())
                                         ? max(1u, (in_img_data->mip_level(0).size().x * 3u) / 4u) * compressed_block_size(in_img_data->format())
                                         : (in_img_data->mip_level(0).size().x * bit_per_pixel(in_img_data->format()) + 7) / 8
                                       );
    dds9_header->dwDepth             = (in_img_data->mip_level(0).size().z > 1 ? in_img_data->mip_level(0).size().z : 0);
    dds9_header->dwMipMapCount       = (in_img_data->mip_level_count() > 1 ? in_img_data->mip_level_count() : 0);
    dds9_header->ddspf.dwSize        = sizeof(DDS_PIXELFORMAT);
    dds9_header->ddspf.dwFlags       = dds_flags(in_img_data->format());
    dds9_header->ddspf.dwFourCC      = dds_fourcc(in_img_data->format());
    dds9_header->ddspf.dwRGBBitCount = (dds9_header->ddspf.dwFourCC & DDPF_FOURCC ? 0 : bit_per_pixel(in_img_data->format()));
    unsigned rm = 0;
    unsigned gm = 0;
    unsigned bm = 0;
    unsigned am = 0;
    if (!dds_bitmask(in_img_data->format(), rm, gm, bm, am)) {
        glerr() << log::error
                << "texture_loader_dds::save_image_data_dx9(): error generating dds header bitmask info." << log::end;
        return false;
    }
    dds9_header->ddspf.dwRBitMask   = rm;
    dds9_header->ddspf.dwGBitMask   = gm;
    dds9_header->ddspf.dwBBitMask   = bm;
    dds9_header->ddspf.dwABitMask   = am;
    dds9_header->dwSurfaceFlags     =   DDSCAPS_TEXTURE
                                      | (in_img_data->mip_level_count() > 1 ? DDSCAPS_MIPMAP : 0)
                                      | (in_img_data->mip_level_count() > 1 ? DDSCAPS_COMPLEX : 0);
    dds9_header->dwSurfaceFlags2    = (in_img_data->mip_level(0).size().z > 1 ? DDSCAPS2_VOLUME : 0);

    if (out_file->write(dds9_header.get(), dds_header_off, sizeof(DDS_HEADER)) != sizeof(DDS_HEADER)) {
        glerr() << log::error
                << "texture_loader_dds::save_image_data_dx9(): error writing to output file: " << in_image_path 
                << " (number of bytes attempted to write: " << sizeof(DDS_HEADER) << ", at position : " << dds_header_off << ")" << log::end;
        return false;
    }

    { // write image data
        io::file::offset_type woff = dds_data_off;
        for (unsigned l = 0; l < static_cast<unsigned>(in_img_data->mip_level_count()); ++l) {
            const scm::size_t         lmip_img_size = mip_level_size(in_img_data->mip_level(l).size(), in_img_data->format());

            if (out_file->write(in_img_data->mip_level(l).data().get(), woff, lmip_img_size) != lmip_img_size) {
                glerr() << log::error
                        << "texture_loader_dds::save_image_data_dx9(): error writing to output file: " << in_image_path 
                        << " (number of bytes attempted to write: " << lmip_img_size << ", at position : " << woff << ")" << log::end;
                return false;
            }

            woff += lmip_img_size;
        }
    }

    out_file->close();

    if (in_img_data->origin() == texture_image_data::ORIGIN_UPPER_LEFT) {
        if (!in_img_data->flip_vertical()) {
            glerr() << log::error
                    << "texture_loader_dds::save_image_data_dx9(): error flipping image data after save operation." << log::end;
            return false;
        }
    }

    return true;
}


texture_2d_ptr
texture_loader_dds::load_texture_2d(render_device&       in_device,
                                    const std::string&   in_image_path) const
{
    texture_image_data_ptr img_data = load_image_data(in_image_path);
    if (!img_data) {
        glerr() << log::error
                << "texture_loader_dds::load_texture_2d(): error opening dds file: " << in_image_path << log::end;
        return texture_2d_ptr();
    }


    std::vector<void*>  image_mip_data_raw;

    for (int i = 0; i < img_data->mip_level_count(); ++i) {
        image_mip_data_raw.push_back(img_data->mip_level(i).data().get());
    }

    texture_2d_ptr new_tex = in_device.create_texture_2d(img_data->mip_level(0).size(), img_data->format(),
                                                         img_data->mip_level_count(), img_data->array_layers(), 1,
                                                         img_data->format(), image_mip_data_raw);

    if (!new_tex) {
        glerr() << log::error << "texture_loader_dds::load_texture_2d(): "
                << "unable to create texture object (file: " << in_image_path << ")" << log::end;
        return texture_2d_ptr();
    }

    image_mip_data_raw.clear();
    img_data.reset();

    return new_tex;
}

texture_3d_ptr
texture_loader_dds::load_texture_3d(render_device&       in_device,
                                    const std::string&   in_image_path) const
{
    texture_image_data_ptr img_data = load_image_data(in_image_path);
    if (!img_data) {
        glerr() << log::error
                << "texture_loader_dds::load_texture_3d(): error opening dds file: " << in_image_path << log::end;
        return texture_3d_ptr();
    }


    std::vector<void*>  image_mip_data_raw;

    for (int i = 0; i < img_data->mip_level_count(); ++i) {
        image_mip_data_raw.push_back(img_data->mip_level(i).data().get());
    }

    texture_3d_ptr new_tex = in_device.create_texture_3d(img_data->mip_level(0).size(), img_data->format(),
                                                         img_data->mip_level_count(),
                                                         img_data->format(), image_mip_data_raw);

    if (!new_tex) {
        glerr() << log::error << "texture_loader_dds::load_texture_3d(): "
                << "unable to create texture object (file: " << in_image_path << ")" << log::end;
        return texture_3d_ptr();
    }

    image_mip_data_raw.clear();
    img_data.reset();

    return new_tex;
}

} // namespace gl
} // namespace scm