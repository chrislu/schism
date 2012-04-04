
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "segy.h"

#include <string.h>

#include <exception>
#include <stdexcept>

#include <scm/core/math.h>
#include <scm/core/io/file.h>
#include <scm/core/platform/byte_swap.h>
#include <scm/core/platform/system_info.h>

namespace {

void
segy_ebcdic_to_ascii(char* s, const scm::size_t ssize)
{
    static unsigned char ebc_to_ascii[] =
        { 0x00, 0x01, 0x02, 0x03, 0x9c, 0x09, 0x86, 0x7f,
          0x97, 0x8d, 0x8e, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
          0x10, 0x11, 0x12, 0x13, 0x9d, 0x85, 0x08, 0x87,
          0x18, 0x19, 0x92, 0x8f, 0x1c, 0x1d, 0x1e, 0x1f,
          0x80, 0x81, 0x82, 0x83, 0x84, 0x0a, 0x17, 0x1b,
          0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x05, 0x06, 0x07,
          0x90, 0x91, 0x16, 0x93, 0x94, 0x95, 0x96, 0x04,
          0x98, 0x99, 0x9a, 0x9b, 0x14, 0x15, 0x9e, 0x1a,
          0x20, 0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6,
          0xa7, 0xa8, 0xd5, 0x2e, 0x3c, 0x28, 0x2b, 0x7c,
          0x26, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
          0xb0, 0xb1, 0x21, 0x24, 0x2a, 0x29, 0x3b, 0x5e,
          0x2d, 0x2f, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
          0xb8, 0xb9, 0xe5, 0x2c, 0x25, 0x5f, 0x3e, 0x3f,
          0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0, 0xc1,
          0xc2, 0x60, 0x3a, 0x23, 0x40, 0x27, 0x3d, 0x22,
          0xc3, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
          0x68, 0x69, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9,
          0xca, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70,
          0x71, 0x72, 0xcb, 0xcc, 0xcd, 0xce, 0xcf, 0xd0,
          0xd1, 0x7e, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
          0x79, 0x7a, 0xd2, 0xd3, 0xd4, 0x5b, 0xd6, 0xd7,
          0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
          0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0x5d, 0xe6, 0xe7,
          0x7b, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
          0x48, 0x49, 0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed,
          0x7d, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50,
          0x51, 0x52, 0xee, 0xef, 0xf0, 0xf1, 0xf2, 0xf3,
          0x5c, 0x9f, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
          0x59, 0x5a, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9,
          0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
          0x38, 0x39, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff };

    unsigned char* s_beg = reinterpret_cast<unsigned char*>(s);
    unsigned char* s_end = (ssize == 0) ? (s_beg + strlen(s)) : s_beg + ssize;

    while (s_beg < s_end) {
        *s_beg = ebc_to_ascii[*s_beg];
        ++s_beg;
    }
}

void
swap_segy_binary_header(scm::gl::data::segy_binary_header& h)
{
    using scm::swap_bytes;

    swap_bytes(&h._job_id);
    swap_bytes(&h._line_num);
    swap_bytes(&h._reel_num);
    swap_bytes(&h._traces_per_ensemble);
    swap_bytes(&h._aux_traces_per_ensemble);
    swap_bytes(&h._sample_interval);
    swap_bytes(&h._sample_interval_orig);
    swap_bytes(&h._samples_per_trace);
    swap_bytes(&h._samples_per_trace_orig);
    swap_bytes(&h._data_format);
    swap_bytes(&h._ensemble_fold);
    swap_bytes(&h._ensemble_type);
    swap_bytes(&h._vert_sum);
    swap_bytes(&h._sweep_freq_beg);
    swap_bytes(&h._sweep_freq_end);
    swap_bytes(&h._sweep_length);
    swap_bytes(&h._sweep_type);
    swap_bytes(&h._sweep_channels);
    swap_bytes(&h._taper_lenght_beg);
    swap_bytes(&h._taper_lenght_end);
    swap_bytes(&h._taper_type);
    swap_bytes(&h._cor_data_traces);
    swap_bytes(&h._bin_gain_recovered);
    swap_bytes(&h._amp_rec_method);
    swap_bytes(&h._measurement_system);
    swap_bytes(&h._signal_polarity);
    swap_bytes(&h._vib_polarity_code);
    swap_bytes(&h._revision);
    swap_bytes(&h._fixed_length_flag);
    swap_bytes(&h._num_ext_headers);
}

void
swap_segy_trace_header(scm::gl::data::segy_trace_header& h)
{
    using scm::swap_bytes;

    swap_bytes(&h._line_sequence_num);
    swap_bytes(&h._file_sequence_num);
    swap_bytes(&h._field_rec_num);
    swap_bytes(&h._trace_num_orig);
    swap_bytes(&h._src_point_num);
    swap_bytes(&h._ensemble_num);
    swap_bytes(&h._traces_per_ensemble);
    swap_bytes(&h._trace_id);
    swap_bytes(&h._num_ver_summed_traces);
    swap_bytes(&h._num_hor_stacked_traces);
    swap_bytes(&h._data_use);
    swap_bytes(&h._dist_src_grp);
    swap_bytes(&h._grp_elev);
    swap_bytes(&h._src_elev);
    swap_bytes(&h._src_depth);
    swap_bytes(&h._grp_datum);
    swap_bytes(&h._src_datum);
    swap_bytes(&h._src_water_depth);
    swap_bytes(&h._grp_water_depth);
    swap_bytes(&h._elev_scalar);
    swap_bytes(&h._coord_scalar);
    swap_bytes(&h._src_coord_x);
    swap_bytes(&h._src_coord_y);
    swap_bytes(&h._grp_coord_x);
    swap_bytes(&h._grp_coord_y);
    swap_bytes(&h._coord_units);
    swap_bytes(&h._weathering_vel);
    swap_bytes(&h._subweathering_vel);
    swap_bytes(&h._src_uphole_time);
    swap_bytes(&h._grp_uphole_time);
    swap_bytes(&h._src_static_cor);
    swap_bytes(&h._grp_static_cor);
    swap_bytes(&h._total_static);
    swap_bytes(&h._lag_time_a);
    swap_bytes(&h._lag_time_b);
    swap_bytes(&h._delay_rec_time);
    swap_bytes(&h._mute_time_beg);
    swap_bytes(&h._mute_time_end);
    swap_bytes(&h._num_samples);
    swap_bytes(&h._sample_interval);
    swap_bytes(&h._gain_type);
    swap_bytes(&h._instrument_gain_const);
    swap_bytes(&h._instrument_gain_initial);
    swap_bytes(&h._correlated);
    swap_bytes(&h._sweep_freq_beg);
    swap_bytes(&h._sweep_freq_end);
    swap_bytes(&h._sweep_length);
    swap_bytes(&h._sweep_type);
    swap_bytes(&h._sweep_taper_beg);
    swap_bytes(&h._sweep_taper_end);
    swap_bytes(&h._taper_type);
    swap_bytes(&h._alias_filter_freq);
    swap_bytes(&h._alias_filter_slope);
    swap_bytes(&h._notch_filter_freq);
    swap_bytes(&h._notch_filter_slope);
    swap_bytes(&h._lo_cut_freq);
    swap_bytes(&h._hi_cut_freq);
    swap_bytes(&h._lo_cut_slope);
    swap_bytes(&h._hi_cut_slope);
    swap_bytes(&h._year);
    swap_bytes(&h._day);
    swap_bytes(&h._hour);
    swap_bytes(&h._min);
    swap_bytes(&h._sec);
    swap_bytes(&h._time_basis_code);
    swap_bytes(&h._weight_factor);
    swap_bytes(&h._geophone_grp_num_switch_1);
    swap_bytes(&h._geophone_grp_num_trace_1);
    swap_bytes(&h._geophone_grp_num_trace_n);
    swap_bytes(&h._gap_size);
    swap_bytes(&h._over_travel);
    swap_bytes(&h._ensemble_cdp_x);
    swap_bytes(&h._ensemble_cdp_y);
    swap_bytes(&h._poststack_inline_num);
    swap_bytes(&h._poststack_crline_num);
    swap_bytes(&h._shotpnt_num);
    swap_bytes(&h._shotpnt_scalar);
    swap_bytes(&h._trace_measurement_unit);
    swap_bytes(&h._transduction_const_mant);
    swap_bytes(&h._transduction_const_exp);
    swap_bytes(&h._transduction_units);
    swap_bytes(&h._device_id);
    swap_bytes(&h._time_scalar);
    swap_bytes(&h._src_type_orient);
    swap_bytes(&h._src_energy_dir0);
    swap_bytes(&h._src_energy_dir);
    swap_bytes(&h._src_measurement_mant);
    swap_bytes(&h._src_measurement_exp);
    swap_bytes(&h._src_measurement_units);
}

}

namespace scm {
namespace gl {
namespace data {
    
segy_trace::segy_trace()
  : _header(new segy_trace_header)
  , _data_offset(0)
{
}

segy_trace::~segy_trace()
{
    _header.reset();
}

segy_data::segy_data(const io::file_ptr& segy_file)
  : _swap_bytes_required(false)
  , _is_ebcdic(false)
  , _volume_size(math::vec3ui(0u))
  , _volume_format(FORMAT_NULL)
  , _traces_start(0)
  , _trace_format(SEGY_FORMAT_NULL)
  , _trace_size(0)
{
    using namespace scm::math;

    if (   !segy_file
        && !segy_file->is_open()) {
        throw std::runtime_error("segy_data::segy_data(): invalid file pointer passed.");
    }

    { // read text header
        _text_header.reset(new segy_text_header);
        if (segy_file->read(_text_header.get(), 0, segy_text_header_size) != segy_text_header_size) {
            throw std::runtime_error("segy_data::segy_data(): error reading segy text header.");
        }
        _is_ebcdic = (_text_header->_text_data[0] != 'C');
        if (_is_ebcdic) {
            segy_ebcdic_to_ascii(_text_header->_text_data, segy_text_header_size);
        }
    }

    { // read binary header
        _binary_header.reset(new segy_binary_header);
        if (segy_file->read(_binary_header.get(), segy_text_header_size, sizeof(segy_binary_header)) != sizeof(segy_binary_header)) {
            throw std::runtime_error("segy_data::segy_data(): error reading segy binary header.");
        }
        // use data format to determine mismatching endianess (data_format <= 8, means msb needs to be 0)
        _swap_bytes_required = (_binary_header->_data_format > 255);
        if (_swap_bytes_required) {
            swap_segy_binary_header(*_binary_header);
        }
    }

    io::offset_type traces_start_offset = 0;
    { // read extened text headers
        io::offset_type next_hdr_offset = segy_text_header_size + sizeof(segy_binary_header);

        if (_binary_header->_num_ext_headers > 0) {
            // read the header data just to be sure we have a valid file
            for (int i = 0; i < _binary_header->_num_ext_headers; ++i) {
                segy_text_header_ptr ext_hdr(new segy_text_header);
                if (segy_file->read(ext_hdr.get(), next_hdr_offset, segy_text_header_size) != segy_text_header_size) {
                    throw std::runtime_error("segy_data::segy_data(): error reading segy extended text header.");
                }

                if (_is_ebcdic) {
                    segy_ebcdic_to_ascii(ext_hdr->_text_data, segy_text_header_size);
                }

                _extended_text_headers.push_back(ext_hdr);
                next_hdr_offset += segy_text_header_size;
            }
        }
        else if (_binary_header->_num_ext_headers < 0) {
            // ok, now we have to read until we encounter the end flag
            while (true) {
                segy_text_header_ptr ext_hdr(new segy_text_header);
                if (segy_file->read(ext_hdr.get(), next_hdr_offset, segy_text_header_size) != segy_text_header_size) {
                    throw std::runtime_error("segy_data::segy_data(): error reading segy extended text header.");
                }

                if (_is_ebcdic) {
                    segy_ebcdic_to_ascii(ext_hdr->_text_data, segy_text_header_size);
                }

                _extended_text_headers.push_back(ext_hdr);
                next_hdr_offset += segy_text_header_size;

                // check for the terminating string
                static const std::string end_token("((SEG: EndText))");
                const std::string cur_hdr(ext_hdr->_text_data, ext_hdr->_text_data + segy_text_header_size);
                if (cur_hdr.find(end_token) != std::string::npos) {
                    break;
                }
            }
        }

        traces_start_offset = next_hdr_offset;
    }

    segy_format     trace_fmt  = to_segy_format(_binary_header->_data_format);
    scm::size_t     trace_size = static_cast<scm::size_t>(_binary_header->_samples_per_trace) * size_of_format(trace_fmt) + sizeof(segy_trace_header);
    scm::size_t     num_traces = (segy_file->size() - traces_start_offset) / trace_size;
    vec3ui          vdim       = vec3ui(0u);

    { // read/parse traces
        io::offset_type next_trace_offset = traces_start_offset;

        segy_format     trace_fmt  = to_segy_format(_binary_header->_data_format);
        scm::size_t     trace_size = static_cast<scm::size_t>(_binary_header->_samples_per_trace) * size_of_format(trace_fmt) + sizeof(segy_trace_header);
        scm::size_t     num_traces = (segy_file->size() - traces_start_offset) / trace_size;

        vdim.x = _binary_header->_samples_per_trace;
        vdim.y = max<int16>(1, _binary_header->_traces_per_ensemble);
        vdim.z = static_cast<uint32>(num_traces / vdim.y);

        scm::size_t     exp_fsize = trace_size * static_cast<scm::size_t>(vdim.y) * vdim.z + traces_start_offset;

        if (vdim.y <= 1 || exp_fsize != segy_file->size()) {
            segy_trace trace_00;
            segy_trace trace_01;
            if (   segy_file->read(trace_00._header.get(), next_trace_offset,              sizeof(segy_trace_header)) != sizeof(segy_trace_header)
                || segy_file->read(trace_01._header.get(), next_trace_offset + trace_size, sizeof(segy_trace_header)) != sizeof(segy_trace_header)) {
                throw std::runtime_error("segy_data::segy_data(): error reading segy first trace headers.");
            }
            if (_swap_bytes_required) {
                swap_segy_trace_header(*trace_00._header);
                swap_segy_trace_header(*trace_01._header);
            }

            bool use_crline_num      = trace_00._header->_poststack_crline_num != trace_01._header->_poststack_crline_num;
            bool use_censemble_cdp_x = trace_00._header->_ensemble_cdp_x != trace_01._header->_ensemble_cdp_x;

            vdim               = vec3ui(0u);
            vdim.x             = trace_00._header->_num_samples;
            int32 trace_00_ens;
            int32 trace_n_ens;

            if (use_crline_num) {
                trace_00_ens = trace_00._header->_poststack_crline_num;
                trace_n_ens  = 0;
            }
            else if (use_censemble_cdp_x) {
                trace_00_ens = trace_00._header->_ensemble_cdp_x;
            }
            else {
                throw std::runtime_error("segy_data::segy_data(): no way to know the volume dimensions here... stupid f***ing SEGY.");
            }

            next_trace_offset += trace_size;

            bool still_match = true;

            do {
                segy_trace trace_n;
                if (segy_file->read(trace_n._header.get(), next_trace_offset, sizeof(segy_trace_header)) != sizeof(segy_trace_header)) {
                    throw std::runtime_error("segy_data::segy_data(): error reading segy trace header.");
                }
                if (_swap_bytes_required) {
                    swap_segy_trace_header(*trace_n._header);
                }
                vdim.y            += 1;
                next_trace_offset += trace_size;

                if (use_crline_num) {
                    trace_n_ens = trace_n._header->_poststack_crline_num;
                }
                else if (use_censemble_cdp_x) {
                    trace_n_ens = trace_n._header->_ensemble_cdp_x;
                }
                exp_fsize   = trace_size * static_cast<scm::size_t>(vdim.y + 1) + traces_start_offset;

                still_match = trace_00_ens != trace_n_ens;
            } while (still_match && static_cast<io::size_type>(exp_fsize) <= segy_file->size());

            vdim.z = static_cast<uint32>(num_traces / vdim.y);
            exp_fsize = trace_size * static_cast<scm::size_t>(vdim.y) * vdim.z + traces_start_offset;

            if (exp_fsize != segy_file->size()) {
                throw std::runtime_error("segy_data::segy_data(): something fishy here, the file size does not check out with the determined volume dimensions.");
            }
        }
    }

    _traces_start  = traces_start_offset;
    _trace_format  = trace_fmt;
    _trace_size    = trace_size;

    _volume_size   = vdim;
    _volume_format = to_gl_format(_trace_format);
}

segy_data::~segy_data()
{
    _extended_text_headers.clear();
    _binary_header.reset();
    _text_header.reset();
}

int
segy_data::size_of_format(segy_format d) const
{
    switch (d) {
        case SEGY_FORMAT_IBM:   return 4;
        case SEGY_FORMAT_INT4:  return 4;
        case SEGY_FORMAT_INT2:  return 2;
        case SEGY_FORMAT_IEEE:  return 4;
        case SEGY_FORMAT_INT1:  return 1;
        default:                return 0;
    }
}

segy_data::segy_format
segy_data::to_segy_format(scm::int16 i) const
{
    switch (i) {
        case 1:     return SEGY_FORMAT_IBM;
        case 2:     return SEGY_FORMAT_INT4;
        case 3:     return SEGY_FORMAT_INT2;
        case 5:     return SEGY_FORMAT_IEEE;
        case 8:     return SEGY_FORMAT_INT1;
        default:    return SEGY_FORMAT_NULL;
    }

}

gl::data_format
segy_data::to_gl_format(segy_format d) const
{
    switch (d) {
        case SEGY_FORMAT_IBM:   return FORMAT_R_32F;
        case SEGY_FORMAT_INT4:  return FORMAT_R_32I;
        case SEGY_FORMAT_INT2:  return FORMAT_R_16;
        case SEGY_FORMAT_IEEE:  return FORMAT_R_32F;
        case SEGY_FORMAT_INT1:  return FORMAT_R_8;
        default:                return FORMAT_NULL;
    }
}

} // namespace data
} // namespace gl
} // namespace scm
