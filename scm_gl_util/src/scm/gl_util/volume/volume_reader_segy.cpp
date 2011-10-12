
#include "volume_reader_segy.h"

#include <exception>
#include <stdexcept>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

#include <scm/core/io/file.h>
#include <scm/core/platform/byte_swap.h>

#include <scm/gl_core/log.h>

#include <scm/gl_util/volume/segy/segy.h>

namespace {

/*
************************************************************
NAME
  r4ibmieee - Convert IBM 370 single precision floating point to
          IEEE representation.
SYNOPSIS
  SUN C
    void r4ibmieee_(f)
    float *f;
  HP9000 C
    void r4ibmieee(f)
    float *f;
  FORTRAN
    call r4ibmieee(f)
    real f
DESCRIPTION
  This routine converts the datum in place: the contents of f are
  changed.  Non-normalized IEEE values are supported.
  Note:  Integer data does not need to be converted.
RETURN VALUE
  None.  f is changed.
ERRORS
  If IBM exponent is too large, f is returned as HUGE, which is defined
  in math.h.
AUTHOR/DATE
  Gordon Maclean  UCLA Inst. of Geophysics, Feb 89
UPDATES
**************************************************************
*/
inline void
r4ibmieee(float* n)
{
    union {
        float val;
        struct ibmfloat {
            unsigned int sign :1;
            unsigned int exp  :7;
            unsigned int mant :24;
        } fields;
        long int ival;
    } ibm;
    union {
        float val;
        struct ieeefloat {
            unsigned int sign :1;
            unsigned int exp  :8;
            unsigned int mant :23;
        } fields;
    } ieee;
    int exp;

 
    ibm.val = *n;
    if (ibm.ival == 0) return;  /* zero is zero in both computers. */
    ieee.val = (float) ibm.fields.mant;

    /* ieee.fields.exp - 24 + 4*(ibm.fields.exp - 64)   */

    exp = ieee.fields.exp + 4 * ibm.fields.exp - 280;
    /*
    Use IEEE un-normalized numbers if exponent is too small.
    Add hidden bit, shift right.        
    */
    if (exp < 0) {
        ieee.fields.mant = (ieee.fields.mant + 0x800000l) >> (-exp);
        ieee.fields.exp = 0;
    }
    else if (exp > 254) {
        ieee.val = static_cast<float>(HUGE);
    }
    else {
        ieee.fields.exp = exp;
    }
    
    ieee.fields.sign = ibm.fields.sign;
    *n = ieee.val;

    return;
}

inline void
ibm_to_ieee(float* f)
{
    register int fc;
    register int fmant;
    register int t;

    fc = *reinterpret_cast<int*>(f);
    if (fc) {
        fmant = 0x00ffffff & fc;
        t = ((0x7f000000 & fc) >> 22) - 130;
        if (t <= 0) {
            fc = 0;
        } else {
            while (!(fmant & 0x00800000) && t != -1)
            {
                --t;
                fmant <<= 1;
            }
            if (t > 254) {
                fc = (0x80000000 & fc) | 0x7f7fffff;
            }
            else if (t <= 0) {
                fc = 0;
            }
            else {
                fc = (0x80000000 & fc) | (t << 23) |(0x007fffff & fmant);
            }
        }
        *f = *reinterpret_cast<float*>(&fc);
    }
}


} // namespace


namespace scm {
namespace gl {

volume_reader_segy::volume_reader_segy(const std::string& file_path,
                                             bool         file_unbuffered)
  : volume_reader(file_path, file_unbuffered)
{
    using namespace boost::filesystem;

    path            fpath(file_path);
    std::string     fname = fpath.filename().string();
    std::string     fext  = fpath.extension().string();

    _file = make_shared<io::file>();
    
    if (!_file->open(fpath.string(), std::ios_base::in, file_unbuffered)) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_segy::volume_reader_segy(): "
                << "error opening volume file (" << fpath.string() << ")." << scm::log::end;
        return;
    }

    try {
        _segy_data = make_shared<data::segy_data>(_file);
        _segy_slice_buffer.reset(new uint8[_segy_data->_trace_size * _segy_data->_volume_size.y]);
    }
    catch (std::exception& e) {
        _file.reset();
        glerr() << scm::log::error
                << "volume_reader_segy::volume_reader_segy(): "
                << "error reading segt file data (" << fpath.string() << "):" << scm::log::nline
                << e.what()
                << scm::log::end;
        return;
    }

    _dimensions = _segy_data->_volume_size;
    _format     = _segy_data->_volume_format;
}

volume_reader_segy::~volume_reader_segy()
{
    _segy_slice_buffer.reset();
    _segy_data.reset();
}

bool
volume_reader_segy::read(const scm::math::vec3ui& o,
                         const scm::math::vec3ui& s,
                               void*              d)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    if (!(*this)) {
        return false;
    }

    {
        // read subvolume
        //if (   (o.x + s.x > _dimensions.x)
        //    || (o.y + s.y > _dimensions.y)
        //    || (o.z + s.z > _dimensions.z)) {
        //    return false;
        //}

        //if (o.x == 0 && s.x == _dimensions.x) {
        //    // we can read complete sets of lines/traces
        //}

        scm::int64 offset_src;
        scm::int64 offset_dst;

        const int64             data_value_size = static_cast<int64>(size_of_format(_format));
        const vec<int64, 3>     o64(o);
        const vec<int64, 3>     d64(_dimensions);
        const vec<int64, 3>     s64(s);
        const vec3ui            read_dim = clamp(s + o, vec3ui(0u), _dimensions) - o;
        const int64             dstart = _segy_data->_traces_start;
        const int64             thsize = sizeof(data::segy_trace_header);

        for (unsigned int s = 0; s < read_dim.z; ++s) {
            for (unsigned int l = 0; l < read_dim.y; ++l) {
                offset_src =  o64.x
                            + d64.x * (o64.y + l)
                            + d64.x * d64.y * (o64.z + s);
                offset_src *= data_value_size;
                offset_src += thsize * ((o64.y + l) + d64.y * (o64.z + s)); // consider the trace headers

                offset_dst =  s64.x * l
                            + s64.x * s64.y * s;
                offset_dst *= data_value_size;

                scm::int64 read_off  = dstart + offset_src;
                scm::int64 read_size = data_value_size * read_dim.x + thsize;


                if (_file->read(_segy_slice_buffer.get(), read_off, read_size) != read_size) {
                    return false;
                }
                else {
                    char* dst_data = reinterpret_cast<char*>(d) + offset_dst;
                    char* src_data = reinterpret_cast<char*>(_segy_slice_buffer.get()) + thsize;

                    if (_segy_data->_swap_bytes_required) {
                        switch (size_of_channel(_format)) {
                            case 1: swap_bytes_array(reinterpret_cast<uint8*>(dst_data),  reinterpret_cast<uint8*>(src_data),  (read_size - thsize) / sizeof(uint8));  break;
                            case 2: swap_bytes_array(reinterpret_cast<uint16*>(dst_data), reinterpret_cast<uint16*>(src_data), (read_size - thsize) / sizeof(uint16)); break;
                            case 4: swap_bytes_array(reinterpret_cast<uint32*>(dst_data), reinterpret_cast<uint32*>(src_data), (read_size - thsize) / sizeof(uint32)); break;
                            case 8: swap_bytes_array(reinterpret_cast<uint64*>(dst_data), reinterpret_cast<uint64*>(src_data), (read_size - thsize) / sizeof(uint64)); break;
                        default:
                            return false;
                        }
                    }
                    else {
                        memcpy(dst_data, src_data, read_size - thsize);
                    }

                    if (_segy_data->_trace_format == data::segy_data::SEGY_FORMAT_IBM) {
                        size_t c = (read_size - thsize) / sizeof(float);
                        float* fp_data = reinterpret_cast<float*>(dst_data);
                        for (scm::size_t i = 0; i < c; ++i) {
                            ibm_to_ieee(fp_data + i);
                            //r4ibmieee(fp_data + i);
                        }
                    }

                }
            }
        }
    }


    return true;
}

} // namespace gl
} // namespace scm
