
#include "volume_data.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/numeric/conversion/bounds.hpp>

#include <scm/log.h>
#include <scm/core/log/logger_state.h>
#include <scm/core/math.h>
#include <scm/core/memory.h>
#include <scm/core/numeric_types.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/data/analysis/transfer_function/build_lookup_table.h>

#include <scm/gl_core/math.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/imaging/texture_data_util.h>
#include <scm/gl_util/primitives/box.h>
#include <scm/gl_util/primitives/box_volume.h>
#include <scm/gl_util/viewer/camera.h>
#include <scm/gl_util/volume/volume_reader_raw.h>
#include <scm/gl_util/volume/volume_reader_segy.h>
#include <scm/gl_util/volume/volume_reader_vgeo.h>

namespace scm {
namespace data {

volume_data::volume_data(const gl::render_device_ptr& device,
                         const std::string&           file_name,
                         const color_map_type&        cmap,
                         const alpha_map_type&        amap)
  : _transform(math::mat4f::identity())
  , _color_map(new color_map_type(cmap))
  , _alpha_map(new alpha_map_type(amap))
  , _selected_lod(0.0f)
{
    using namespace scm::gl;
    using namespace scm::math;

    out() << log::info << "volume_data::volume_data(): loading raw volume..." << log::end;
    _volume_raw = load_volume(device, file_name);
    if (!_volume_raw) {
        throw std::runtime_error("volume_data::volume_data(): error loading volume data from file: " + file_name);
    }
    out() << log::info << "volume_data::volume_data(): loading raw volume done." << log::end;

    out() << log::info << "volume_data::volume_data(): generating color map..." << log::end;
    _color_alpha_map = create_color_alpha_map(device, 256);
    if (!_color_alpha_map) {
        throw std::runtime_error("volume_data::volume_data(): error creating color alpha map texture.");
    }
    if (!update_color_alpha_map(device->main_context())) {
        throw std::runtime_error("volume_data::volume_data(): error upating color alpha map texture.");
    }
    _color_alpha_map_dirty = false;
    out() << log::info << "volume_data::volume_data(): generating color map done." << log::end;

    _data_dimensions = _volume_raw->descriptor()._size;
    _max_lod         = static_cast<float>(gl::util::max_mip_levels(_data_dimensions)) - 1.0f;
    unsigned max_dim = max(max(_data_dimensions.x,
                               _data_dimensions.y),
                               _data_dimensions.z);
    _extends         = vec3f(_volume_raw->descriptor()._size) / max_dim;
    _bbox            = box(math::vec3f(0.0f), _extends);
    _transform       = make_translation(-_extends / 2.0f);

    out() << log::info << "volume_data::volume_data(): generating render resources..." << log::end;
    try {
        _bbox_geometry = make_shared<box_volume_geometry>(device, _bbox.min_vertex(), _bbox.max_vertex());
        _volume_block  = make_uniform_block<volume_uniform_data>(device);
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("volume_data::volume_data(): error creating geometry objects: ") + e.what());
    }
    out() << log::info << "volume_data::volume_data(): generating render resources done." << log::end;

    out() << log::info << "volume_data::volume_data(): successfully loaded volume file: " << file_name << log::end;

    sample_distance_factor(0.5f);
    sample_distance_ref_factor(0.5f);
}

volume_data::~volume_data()
{
    _bbox_geometry.reset();

    _volume_raw.reset();
    _color_alpha_map.reset();

    _volume_block.reset();
}

const math::vec3f&
volume_data::extends() const
{
    return _extends;
}

const math::vec3ui&
volume_data::data_dimensions() const
{
    return _data_dimensions;
}

const math::mat4f&
volume_data::transform() const
{
    return _transform;
}

void
volume_data::transform(const math::mat4f& m)
{
    _transform = m;
}

float
volume_data::sample_distance() const
{
    return _sample_distance;
}

void
volume_data::sample_distance_factor(float d)
{
    using namespace scm::math;

    unsigned max_dim = max(max(_data_dimensions.x,
                               _data_dimensions.y),
                               _data_dimensions.z);

    _sample_distance = d / max_dim;
}

float
volume_data::sample_distance_factor() const
{
    using namespace scm::math;

    unsigned max_dim = max(max(_data_dimensions.x,
                               _data_dimensions.y),
                               _data_dimensions.z);

    return _sample_distance * max_dim;
}

float
volume_data::sample_distance_ref() const
{
    return _sample_distance_ref;
}

void
volume_data::sample_distance_ref_factor(float d)
{
    using namespace scm::math;

    unsigned max_dim = max(max(_data_dimensions.x,
                               _data_dimensions.y),
                               _data_dimensions.z);

    _sample_distance_ref = d / max_dim;
}

float
volume_data::sample_distance_ref_factor() const
{
    using namespace scm::math;

    unsigned max_dim = max(max(_data_dimensions.x,
                               _data_dimensions.y),
                               _data_dimensions.z);

    return _sample_distance_ref * max_dim;
}

float
volume_data::min_value() const
{
    return _min_value;
}

float
volume_data::max_value() const
{
    return _max_value;
}

float
volume_data::selected_lod() const
{
    return _selected_lod;
}

void
volume_data::selected_lod(float l)
{
    _selected_lod = math::clamp(l, 0.0f, _max_lod);
}

const gl::box&
volume_data::bbox() const
{
    return _bbox;
}

const gl::box_volume_geometry_ptr&
volume_data::bbox_geometry() const
{
    return _bbox_geometry;
}

const gl::texture_3d_ptr&
volume_data::volume_raw() const
{
    return _volume_raw;
}

const gl::texture_1d_ptr&
volume_data::color_alpha_map() const
{
    return _color_alpha_map;
}

const volume_data::color_map_ptr&
volume_data::color_map() const
{
    return _color_map;
}

const volume_data::alpha_map_ptr&
volume_data::alpha_map() const
{
    return _alpha_map;
}

void
volume_data::update_color_alpha_maps()
{
    _color_alpha_map_dirty = true;
}

gl::texture_3d_ptr
volume_data::load_volume(const gl::render_device_ptr& in_device,
                         const std::string&           in_file_name)
{
    using namespace scm::gl;
    using namespace scm::math;
    using namespace boost::filesystem;
   
    path                    file_path(in_file_name);
    std::string             file_name       = file_path.filename().string();
    std::string             file_extension  = file_path.extension().string();
    
    boost::algorithm::to_lower(file_extension);

    vec3ui      data_dimensions = vec3ui(0u);
    data_format data_format     = FORMAT_NULL;
    scm::shared_array<unsigned char> read_buffer;
    scm::size_t                      read_buffer_size = 0;

    scoped_ptr<gl::volume_reader> vol_reader;

    out() << log::indent;
    time::high_res_timer timer;

    if (file_extension == ".raw") {
        vol_reader.reset(new volume_reader_raw(file_path.string(), false));
    }
    else if (file_extension == ".vol") {
        vol_reader.reset(new volume_reader_vgeo(file_path.string(), true));
    }
    else if (file_extension == ".segy" || file_extension == ".sgy") {
        vol_reader.reset(new volume_reader_segy(file_path.string(), true));
    }
    else {
        err() << log::error
              << "volume_data::load_volume(): unable to open file ('" << in_file_name << "')." << log::end;
        return texture_3d_ptr();
    }

    if (!(*vol_reader)) {
        err() << log::error
                << "volume_data::load_volume(): unable to open file ('" << in_file_name << "')." << log::end;
        return texture_3d_ptr();
    }
    out() << "source data dimensions: " << vol_reader->dimensions() << log::end;

#if 0
    vec3ui data_offset = vec3ui(0u, vol_reader->dimensions().y / 2, vol_reader->dimensions().z / 2);//vol_reader->dimensions() / vec3ui(2, 2, 2);
    data_dimensions    = vol_reader->dimensions() / vec3ui(1, 20, 10);
#else
    vec3ui data_offset = vec3ui(0);
    data_dimensions    = vol_reader->dimensions();
#endif
    //data_dimensions.x  = 2048;
    data_format        = vol_reader->format();

    read_buffer_size =   static_cast<scm::size_t>(data_dimensions.x) * data_dimensions.y * data_dimensions.z
                        * size_of_format(data_format);

    read_buffer.reset(new unsigned char[read_buffer_size]);


    out() << "reading volume data "
          << "(dimensions: " << data_dimensions
          << ", size : " << std::fixed << std::setprecision(3) << static_cast<double>(read_buffer_size) / (1024.0*1024.0) << "MiB)..."
          << log::end;
    timer.start();
    if (!vol_reader->read(data_offset, data_dimensions, read_buffer.get())) {
        err() << log::error
                << "volume_data::load_volume(): unable to read data from file ('" << in_file_name << "')." << log::end;
        return texture_3d_ptr();
    }
    timer.stop();
    out() << "reading volume data done"
          << " (elapsed time: " << std::fixed << std::setprecision(3)
          << time::to_seconds(timer.get_time()) << "s, "
          << (static_cast<double>(read_buffer_size) / (1024.0*1024.0)) / time::to_seconds(timer.get_time()) << "MiB/s)" << log::end;

    _min_value = 0.0f;
    _max_value = 1.0f;
    if (is_float_type(data_format)) {
        out() << "determining floating point value range..." << log::end;
        timer.start();

        float* fp_data = reinterpret_cast<float*>(read_buffer.get());
        size_t dcount  = static_cast<scm::size_t>(data_dimensions.x) * data_dimensions.y * data_dimensions.z * channel_count(data_format);
        _min_value = boost::numeric::bounds<float>::highest();//(std::numeric_limits<float>::max)();
        _max_value = boost::numeric::bounds<float>::lowest();//(std::numeric_limits<float>::min)();

        for (size_t i = 0; i < dcount; ++i) {
            _min_value = min(_min_value, fp_data[i]);
            _max_value = max(_max_value, fp_data[i]);
        }

        if (abs(_min_value) > abs(_max_value)) {
            _max_value = abs(_min_value);
        }
        else {
            _min_value = -abs(_max_value);
        }

        timer.stop();
        out() << "etermining floating point value range"
              << " (elapsed time: " << std::fixed << std::setprecision(3)
              << time::to_seconds(timer.get_time()) << "s)" << log::end;

    }
    out() << "min_value: " << _min_value << ", max_value: " << _max_value << log::end;

    std::vector<uint8*> mip_data;
    std::vector<void*>  mip_init_data;

    out() << "generating mip map hierarchy..." << log::end;
    timer.start();
    gl::util::generate_mipmaps(data_dimensions, data_format, read_buffer.get(), mip_data);
    timer.stop();
    out() << "generating mip map hierarchy done"
          << " (elapsed time: " << std::fixed << std::setprecision(3)
          << time::to_seconds(timer.get_time()) << "s)" << log::end;

    std::for_each(mip_data.begin(), mip_data.end(), [&mip_init_data](uint8* v) {mip_init_data.push_back(v);});
    //for (std::vector<uint8*>::iterator v = mip_data.begin(); v != mip_data.end(); ++v) {
    //    mip_init_data.push_back(*v);
    //}

    unsigned mip_count = gl::util::max_mip_levels(data_dimensions);

    assert(mip_count == mip_init_data.size());

    out() << "allocating texture storage ("
          << "dimensions: " << data_dimensions << ", format: " << format_string(data_format)
          << ", mip-level: " << mip_count
          << ", size : " << std::fixed << std::setprecision(3) << static_cast<double>(read_buffer_size) / (1024.0*1024.0) << "MiB)..."
          << log::end;
    timer.start();
    texture_3d_ptr new_volume_tex = in_device->create_texture_3d(data_dimensions, data_format, mip_count, data_format, mip_init_data);
    timer.stop();
    out() << "allocating texture storage done."
          << " (elapsed time: " << std::fixed << std::setprecision(3)
          << time::to_seconds(timer.get_time()) << "s)" << log::end;

    std::for_each(mip_data.begin() + 1, mip_data.end(), [&mip_init_data](uint8* v) {delete [] v; });

    out() << log::outdent;

    return new_volume_tex;
}

gl::texture_1d_ptr
volume_data::create_color_alpha_map(const gl::render_device_ptr& in_device,
                                          unsigned               in_size) const
{
    using namespace scm::gl;
    using namespace scm::math;

    log::logger_format_saver out_save(out().associated_logger());

    out() << log::indent;
    time::high_res_timer timer;

    out() << "allocating texture storage ("
          << "dimensions: " << in_size << ", format: " << format_string(FORMAT_RGBA_8)
          << ", mip-level: " << 1
          << ", size : " << std::fixed << std::setprecision(3) << static_cast<double>(in_size * size_of_format(FORMAT_RGBA_8)) / (1024.0) << "KiB)..."
          << log::end;
    timer.start();
    texture_1d_ptr new_tex = in_device->create_texture_1d(in_size, FORMAT_RGBA_8, 1, 1);
    timer.stop();
    out() << "allocating texture storage done."
          << " (elapsed time: " << std::fixed << std::setprecision(3)
          << time::to_seconds(timer.get_time()) << "s)" << log::end;

    if (!new_tex) {
        err() << log::error
              << "volume_data::create_color_map(): error during color map texture generation." << log::end;
        return texture_1d_ptr();
    }

    return new_tex;
}

bool
volume_data::update_color_alpha_map(const gl::render_context_ptr& context) const
{
    using namespace scm::gl;
    using namespace scm::math;

    scm::scoped_array<scm::math::vec3f>  color_lut;
    scm::scoped_array<float>             alpha_lut;

    log::logger_format_saver out_save(out().associated_logger());

    out() << log::indent;
    time::high_res_timer timer;

    out() << "generating color map texture data..." << log::end;
    timer.start();

    unsigned in_size = _color_alpha_map->descriptor()._size;

    color_lut.reset(new vec3f[in_size]);
    alpha_lut.reset(new float[in_size]);

    if (   !scm::data::build_lookup_table(color_lut, *_color_map, in_size)
        || !scm::data::build_lookup_table(alpha_lut, *_alpha_map, in_size)) {
        err() << log::error
              << "volume_data::update_color_alpha_map(): error during lookuptable generation" << log::end;
        return false;
    }
    scm::scoped_array<float> combined_lut;

    combined_lut.reset(new float[in_size * 4]);

    for (unsigned i = 0; i < in_size; ++i) {
        combined_lut[i*4   ] = color_lut[i].x;
        combined_lut[i*4 +1] = color_lut[i].y;
        combined_lut[i*4 +2] = color_lut[i].z;
        combined_lut[i*4 +3] = alpha_lut[i];
    }
    timer.stop();
    out() << "generating color map texture data done."
          << " (elapsed time: " << std::fixed << std::setprecision(3)
          << time::to_seconds(timer.get_time()) << "s)" << log::end;
    
    out() << "uploading texture data ("
          << "dimensions: " << in_size << ", format: " << format_string(FORMAT_RGBA_8)
          << ", mip-level: " << 1
          << ", size : " << std::fixed << std::setprecision(3) << static_cast<double>(in_size * size_of_format(FORMAT_RGBA_8)) / (1024.0) << "KiB)..."
          << log::end;
    timer.start();


    texture_region ur(vec3ui(0u), vec3ui(in_size, 1, 1));
    bool res = context->update_sub_texture(_color_alpha_map, ur, 0u, FORMAT_RGBA_32F, combined_lut.get());
    timer.stop();
    out() << "uploading texture data done."
          << " (elapsed time: " << std::fixed << std::setprecision(3)
          << time::to_seconds(timer.get_time()) << "s)" << log::end;

    if (!res) {
        err() << log::error
              << "volume_data::update_color_alpha_map(): error during color map texture generation." << log::end;
        return false;
    }

    return true;
}

void
volume_data::update(const gl::render_context_ptr& context,
                    const gl::camera&             cam)
{
    using namespace scm::gl;
    using namespace scm::math;

    float max_dim = static_cast<float>(max(max(_data_dimensions.x,
                                               _data_dimensions.y),
                                               _data_dimensions.z));

    mat4f mv_matrix     = cam.view_matrix() * transform();
    mat4f mv_matrix_inv = inverse(mv_matrix);

    _volume_block.begin_manipulation(context); {
        _volume_block->_volume_extends              = vec4f(extends(), 0.0);
        _volume_block->_scale_obj_to_tex            = vec4f(1.0f) / vec4f(extends(), 1.0);
        _volume_block->_sampling_distance           = vec4f(sample_distance(), sample_distance() / sample_distance_ref(), 0.0, 0.0);
        _volume_block->_os_camera_position          = mv_matrix_inv.column(3) / mv_matrix_inv.column(3).w;
        _volume_block->_value_range                 = vec4f(min_value(), max_value(), max_value() - min_value(), 1.0f / (max_value() - min_value()));

        _volume_block->_m_matrix                     = transform();
        _volume_block->_m_matrix_inverse             = inverse(transform());
        _volume_block->_m_matrix_inverse_transpose   = transpose(_volume_block->_m_matrix_inverse);
        _volume_block->_mv_matrix                    = mv_matrix;
        _volume_block->_mv_matrix_inverse            = mv_matrix_inv;
        _volume_block->_mv_matrix_inverse_transpose  = transpose(mv_matrix_inv);
        _volume_block->_mvp_matrix                   = cam.projection_matrix() * mv_matrix;
        _volume_block->_mvp_matrix_inverse           = inverse(_volume_block->_mvp_matrix);
    } _volume_block.end_manipulation();

    if (_color_alpha_map_dirty) {
        update_color_alpha_map(context);
        _color_alpha_map_dirty = false;
    }
}

const volume_data::volume_uniform_block&
volume_data::volume_block() const
{
    return _volume_block;
}
} // namespace data
} // namespace scm
