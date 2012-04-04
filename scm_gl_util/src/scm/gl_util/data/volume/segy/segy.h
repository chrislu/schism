
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_SEGY_H_INCLUDED
#define SCM_GL_UTIL_SEGY_H_INCLUDED

#include <vector>

#include <scm/core/math.h>
#include <scm/core/memory.h>
#include <scm/core/numeric_types.h>
#include <scm/core/io/io_fwd.h>

#include <scm/gl_core/data_formats.h>

namespace scm {
namespace gl {
namespace data {

// SEGY rev. 1 Data Exchange format
// http://www.freeusp.org/RaceCarWebsite/TheToolkit/FAQ/Bridge/seg_y_rev1.pdf

// SEGY text header, 3200bytes
const scm::size_t   segy_text_header_size = 3200;
struct segy_text_header
{
    char            _text_data[segy_text_header_size];
}; // segy_text_header

// SEGY binary header (starting after text header), 400bytes
struct segy_binary_header
{
    scm::int32  _job_id;                    // bytes 3201 - 3204
    scm::int32  _line_num;                  // bytes 3205 - 3208
    scm::int32  _reel_num;                  // bytes 3209 - 3212
    scm::int16  _traces_per_ensemble;       // bytes 3213 - 3214
    scm::int16  _aux_traces_per_ensemble;   // bytes 3215 - 3216
    scm::int16  _sample_interval;           // (us) bytes 3217 - 3218
    scm::int16  _sample_interval_orig;      // (us) bytes 3219 - 3220
    scm::int16  _samples_per_trace;         // bytes 3221 - 3222
    scm::int16  _samples_per_trace_orig;    // bytes 3223 - 3224
    scm::int16  _data_format;               // bytes 3225 - 3226
    scm::int16  _ensemble_fold;             // bytes 3227 - 3228
    scm::int16  _ensemble_type;             // bytes 3229 - 3230
    scm::int16  _vert_sum;                  // bytes 3231 - 3232
    scm::int16  _sweep_freq_beg;            // bytes 3233 - 3234
    scm::int16  _sweep_freq_end;            // bytes 3235 - 3236
    scm::int16  _sweep_length;              // (ms) bytes 3237 - 3238
    scm::int16  _sweep_type;                // bytes 3239 - 3240
    scm::int16  _sweep_channels;            // bytes 3241 - 3242
    scm::int16  _taper_lenght_beg;          // bytes 3243 - 3244
    scm::int16  _taper_lenght_end;          // bytes 3245 - 3246
    scm::int16  _taper_type;                // bytes 3247 - 3248
    scm::int16  _cor_data_traces;           // bytes 3249 - 3250
    scm::int16  _bin_gain_recovered;        // bytes 3251 - 3252
    scm::int16  _amp_rec_method;            // bytes 3253 - 3254
    scm::int16  _measurement_system;        // bytes 3255 - 3256
    scm::int16  _signal_polarity;           // bytes 3257 - 3258
    scm::int16  _vib_polarity_code;         // bytes 3259 - 3260
    scm::int8   _padding_00[240];           // bytes 3261 - 3500
    scm::int16  _revision;                  // bytes 3501 - 3502
    scm::int16  _fixed_length_flag;         // bytes 3503 - 3504
    scm::int16  _num_ext_headers;           // bytes 3505 - 3506
    scm::int8   _padding_01[94];            // bytes 3507 - 3600
}; // struct segy_binary_header

// SEGY trace hader, 240bytes
struct segy_trace_header
{
    scm::int32  _line_sequence_num;         // bytes 1-4 
    scm::int32  _file_sequence_num;         // bytes 5-8 
    scm::int32  _field_rec_num;             // bytes 9-12
    scm::int32  _trace_num_orig;            // bytes 13-16
    scm::int32  _src_point_num;             // bytes 17-20
    scm::int32  _ensemble_num;              // bytes 21-24
    scm::int32  _traces_per_ensemble;       // bytes 25-28
    scm::int16  _trace_id;                  // bytes 29-30
    scm::int16  _num_ver_summed_traces;     // bytes 31-32
    scm::int16  _num_hor_stacked_traces;    // bytes 33-34
    scm::int16  _data_use;                  // bytes 35-36
    scm::int32  _dist_src_grp;              // bytes 37-40
    scm::int32  _grp_elev;                  // bytes 41-44
    scm::int32  _src_elev;                  // bytes 45-48
    scm::int32  _src_depth;                 // bytes 49-52
    scm::int32  _grp_datum;                 // bytes 53-56
    scm::int32  _src_datum;                 // bytes 57-60
    scm::int32  _src_water_depth;           // bytes 61-64
    scm::int32  _grp_water_depth;           // bytes 65-68
    scm::int16  _elev_scalar;               // bytes 69-70
    scm::int16  _coord_scalar;              // bytes 71-72
    scm::int32  _src_coord_x;               // bytes 73-76
    scm::int32  _src_coord_y;               // bytes 77-80
    scm::int32  _grp_coord_x;               // bytes 81-84
    scm::int32  _grp_coord_y;               // bytes 85-88
    scm::int16  _coord_units;               // bytes 89-90
    scm::int16  _weathering_vel;            // bytes 91-92
    scm::int16  _subweathering_vel;         // bytes 93-94
    scm::int16  _src_uphole_time;           // bytes 95-96
    scm::int16  _grp_uphole_time;           // bytes 97-98
    scm::int16  _src_static_cor;            // bytes 99-100
    scm::int16  _grp_static_cor;            // bytes 101-102
    scm::int16  _total_static;              // bytes 103-104
    scm::int16  _lag_time_a;                // bytes 105-106
    scm::int16  _lag_time_b;                // bytes 107-108
    scm::int16  _delay_rec_time;            // bytes 109-110
    scm::int16  _mute_time_beg;             // bytes 111-112
    scm::int16  _mute_time_end;             // bytes 113-114
    scm::int16  _num_samples;               // bytes 115-116
    scm::int16  _sample_interval;           // bytes 117-118
    scm::int16  _gain_type;                 // bytes 119-120
    scm::int16  _instrument_gain_const;     // bytes 121-122
    scm::int16  _instrument_gain_initial;   // bytes 123-124
    scm::int16  _correlated;                // bytes 125-126
    scm::int16  _sweep_freq_beg;            // bytes 127-128
    scm::int16  _sweep_freq_end;            // bytes 129-130
    scm::int16  _sweep_length;              // bytes 131-132
    scm::int16  _sweep_type;                // bytes 133-134
    scm::int16  _sweep_taper_beg;           // bytes 135-136
    scm::int16  _sweep_taper_end;           // bytes 137-138
    scm::int16  _taper_type;                // bytes 139-140
    scm::int16  _alias_filter_freq;         // bytes 141-142
    scm::int16  _alias_filter_slope;        // bytes 143-144
    scm::int16  _notch_filter_freq;         // bytes 145-146
    scm::int16  _notch_filter_slope;        // bytes 147-148
    scm::int16  _lo_cut_freq;               // bytes 149-150
    scm::int16  _hi_cut_freq;               // bytes 141-152
    scm::int16  _lo_cut_slope;              // bytes 153-154
    scm::int16  _hi_cut_slope;              // bytes 155-156
    scm::int16  _year;                      // bytes 157-158
    scm::int16  _day;                       // bytes 159-160
    scm::int16  _hour;                      // bytes 161-162
    scm::int16  _min;                       // bytes 163-164
    scm::int16  _sec;                       // bytes 165-166
    scm::int16  _time_basis_code;           // bytes 167-168
    scm::int16  _weight_factor;             // bytes 169-170
    scm::int16  _geophone_grp_num_switch_1; // bytes 171-172
    scm::int16  _geophone_grp_num_trace_1;  // bytes 173-174
    scm::int16  _geophone_grp_num_trace_n;  // bytes 175-176
    scm::int16  _gap_size;                  // bytes 177-178
    scm::int16  _over_travel;               // bytes 179-180
    scm::int32  _ensemble_cdp_x;            // bytes 181-184
    scm::int32  _ensemble_cdp_y;            // bytes 185-188
    scm::int32  _poststack_inline_num;      // bytes 189-192
    scm::int32  _poststack_crline_num;      // bytes 193-196
    scm::int32  _shotpnt_num;               // bytes 197-200
    scm::int16  _shotpnt_scalar;            // bytes 201-202
    scm::int16  _trace_measurement_unit;    // bytes 203-204
    scm::int32  _transduction_const_mant;   // bytes 205-208
    scm::int16  _transduction_const_exp;    // bytes 209-210
    scm::int16  _transduction_units;        // bytes 211-212
    scm::int16  _device_id;                 // bytes 213-214
    scm::int16  _time_scalar;               // bytes 215-216
    scm::int16  _src_type_orient;           // bytes 217-218
    scm::int16  _src_energy_dir0;           // bytes 219-220
    scm::int32  _src_energy_dir;            // bytes 221-224
    scm::int32  _src_measurement_mant;      // bytes 225-228
    scm::int16  _src_measurement_exp;       // bytes 229-230
    scm::int16  _src_measurement_units;     // bytes 231-232
    scm::int8   _padding_00[8];             // bytes 233-240
}; // struct segy_trace_header

typedef scm::shared_ptr<segy_text_header>   segy_text_header_ptr;
typedef scm::shared_ptr<segy_binary_header> segy_binary_header_ptr;
typedef scm::shared_ptr<segy_trace_header>  segy_trace_header_ptr;

// segy trace (header, data)
struct segy_trace
{
    segy_trace_header_ptr   _header;
    scm::io::offset_type    _data_offset;

    segy_trace();
    ~segy_trace();
}; // struct segy_trace

// segy data
struct segy_data
{
    enum segy_format {
        SEGY_FORMAT_NULL    = 0,
        SEGY_FORMAT_IBM     = 1,
        SEGY_FORMAT_INT4    = 2,
        SEGY_FORMAT_INT2    = 3,
        SEGY_FORMAT_IEEE    = 5,
        SEGY_FORMAT_INT1    = 8
    };

    typedef std::vector<segy_text_header_ptr>   segy_text_header_ptr_vec;
    typedef std::vector<segy_trace>             segy_trace_vec;

    segy_text_header_ptr        _text_header;
    segy_binary_header_ptr      _binary_header;
    segy_text_header_ptr_vec    _extended_text_headers;

    math::vec3ui                _volume_size;
    gl::data_format             _volume_format;

    io::offset_type             _traces_start;
    segy_format                 _trace_format;
    scm::size_t                 _trace_size;

    bool                        _swap_bytes_required;
    bool                        _is_ebcdic;

    segy_data(const io::file_ptr& segy_file);
    ~segy_data();

    int                         size_of_format(segy_format d) const;
    segy_format                 to_segy_format(scm::int16 i) const;
    gl::data_format             to_gl_format(segy_format d) const;

    //bool        swap_bytes;
    //bool        is_ebcdic;
}; // struct segy_data

} // namespace data
} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_SEGY_H_INCLUDED
