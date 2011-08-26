
#include "texture_loader_dds.h"

#include <cassert>
#include <exception>
#include <stdexcept>
#include <string>

#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>
#include <scm/core/io/file.h>

#include <scm/gl_core/log.h>

#include <scm/gl_util/imaging/texture_image_data.h>

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

    const scm::shared_ptr<const scm::io::file>      file() const              { assert(_file);                  return _file; }
    const scm::shared_ptr<const DDS_HEADER>         dds_header() const        { assert(_dds_header);            return _dds_header; }
    const scm::shared_ptr<const DDS_HEADER_DXT10>   dds_header_dxt10() const  { assert(_dds_header_dxt10);      return _dds_header_dxt10; }
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

} // namespace

namespace scm {
namespace gl {

texture_image_data_ptr
texture_loader_dds::load_image_data(const std::string&  in_image_path)
{
    ::dds_file   raw_dds = ::dds_file(in_image_path);

    if (!raw_dds) {
        glerr() << log::error
                << "texture_loader_dds::load_image_data(): error reading dds image data." << log::end;
        return texture_image_data_ptr();
    }

    return texture_image_data_ptr();
}


} // namespace gl
} // namespace scm