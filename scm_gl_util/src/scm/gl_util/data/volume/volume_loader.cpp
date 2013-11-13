
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_loader.h"

#include <vector>
#include <memory.h>
#include <sstream>

#include <FreeImagePlus.h>

#include <scm/core/math.h>
#include <scm/core/memory.h>


#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/numeric/conversion/bounds.hpp>

#include <scm/log.h>
#include <scm/core/log/logger_state.h>
#include <scm/core/math.h>
#include <scm/core/memory.h>
#include <scm/core/numeric_types.h>
#include <scm/core/time/high_res_timer.h>

#include <scm/gl_util/data/analysis/transfer_function/build_lookup_table.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/math.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/state_objects.h>
#include <scm/gl_core/texture_objects.h>
#include <scm/gl_core/buffer_objects/scoped_buffer_map.h>

#include <scm/gl_util/data/imaging/texture_data_util.h>
#include <scm/gl_util/primitives/box.h>
#include <scm/gl_util/primitives/box_volume.h>
#include <scm/gl_util/viewer/camera.h>
#include <scm/gl_util/data/volume/volume_reader_raw.h>
#include <scm/gl_util/data/volume/volume_reader_segy.h>
#include <scm/gl_util/data/volume/volume_reader_vgeo.h>

#include <scm/gl_util/data/imaging/texture_image_data.h>

namespace scm {
namespace gl {

texture_3d_ptr
volume_loader::load_texture_3d(render_device&       in_device,
                                const std::string&   in_image_path,
                                bool                 in_create_mips,
                                bool                 in_color_mips,
                                const data_format    in_force_internal_format)
{
 
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using namespace boost::filesystem;

    scoped_ptr<gl::volume_reader> vol_reader;
    path                    file_path(in_image_path);
    std::string             file_name       = file_path.filename().string();
    std::string             file_extension  = file_path.extension().string();

    data_format volume_data_format     = FORMAT_NULL;

    boost::algorithm::to_lower(file_extension);

    if (file_extension == ".raw") {
        vol_reader.reset(new scm::gl::volume_reader_raw(file_path.string(), false));
    }
    else if (file_extension == ".vol") {
        vol_reader.reset(new scm::gl::volume_reader_vgeo(file_path.string(), true));
    }
    else {
        err() << log::error
              << "volume_loader::load_texture_3d(): unsupported volume file format ('" << file_extension << "')." << log::end;
        return (texture_3d_ptr());
    }

    if (!(*vol_reader)) {
        std::cout << "volume_loader::load_texture_3d(): unable to open file ('" << in_image_path << "')." << log::end;
        return (texture_3d_ptr());
    }

    int    max_volume_dim  = in_device.capabilities()._max_texture_3d_size;
    vec3ui data_offset = vec3ui(0);
    vec3ui data_dimensions = vol_reader->dimensions();
    volume_data_format     = vol_reader->format();

    if (max(max(data_dimensions.x, data_dimensions.y), data_dimensions.z) > static_cast<unsigned>(max_volume_dim)) {
        err() << log::error
              << "volume_loader::load_texture_3d(): volume too large to load as single texture ('" << data_dimensions << "')." << log::end;
        return (texture_3d_ptr());
    }

    scm::shared_array<unsigned char>    read_buffer;
    scm::size_t                         read_buffer_size =   data_dimensions.x * data_dimensions.y * data_dimensions.z
                                                           * size_of_format(volume_data_format);

    read_buffer.reset(new unsigned char[read_buffer_size]);

    if (!vol_reader->read(data_offset, data_dimensions, read_buffer.get())) {
        err() << log::error
              << "volume_loader::load_texture_3d(): unable to read data from file ('" << in_image_path << "')." << log::end;
        return (texture_3d_ptr());
    }

    if (volume_data_format == FORMAT_NULL) {
        err() << log::error
              << "volume_loader::load_texture_3d(): unable to determine volume data format ('" << in_image_path << "')." << log::end;
        return (texture_3d_ptr());
    }

    std::vector<void*> in_data;
    in_data.push_back(read_buffer.get());
    texture_3d_ptr new_volume_tex =
        in_device.create_texture_3d(data_dimensions, volume_data_format, 1, volume_data_format, in_data);

    return (new_volume_tex);
}

texture_3d_ptr
volume_loader::load_volume_data(render_device&      in_device,
								const std::string&  in_image_path)
{
    using namespace scm::gl;
    using namespace scm::math;
    using namespace boost::filesystem;
   
    path                    file_path(in_image_path);
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
              << "volume_data::load_volume(): unable to open file ('" << in_image_path << "')." << log::end;
        return texture_3d_ptr();
    }

    if (!(*vol_reader)) {
        err() << log::error
                << "volume_data::load_volume(): unable to open file ('" << in_image_path << "')." << log::end;
        return texture_3d_ptr();
    }
    out() << "source data dimensions: " << vol_reader->dimensions() << log::end;

    vec3ui data_offset = vec3ui(0);
    data_dimensions    = vol_reader->dimensions();
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
                << "volume_data::load_volume(): unable to read data from file ('" << in_image_path << "')." << log::end;
        return texture_3d_ptr();
    }
    timer.stop();
    out() << "reading volume data done"
          << " (elapsed time: " << std::fixed << std::setprecision(3)
          << time::to_seconds(timer.get_time()) << "s, "
          << (static_cast<double>(read_buffer_size) / (1024.0*1024.0)) / time::to_seconds(timer.get_time()) << "MiB/s)" << log::end;

    //_min_value = 0.0f;
    //_max_value = 1.0f;
    //if (is_float_type(data_format)) {
    //    out() << "determining floating point value range..." << log::end;
    //    timer.start();

    //    float* fp_data = reinterpret_cast<float*>(read_buffer.get());
    //    size_t dcount  = static_cast<scm::size_t>(data_dimensions.x) * data_dimensions.y * data_dimensions.z * channel_count(data_format);
    //    _min_value = boost::numeric::bounds<float>::highest();//(std::numeric_limits<float>::max)();
    //    _max_value = boost::numeric::bounds<float>::lowest();//(std::numeric_limits<float>::min)();

    //    for (size_t i = 0; i < dcount; ++i) {
    //        _min_value = min(_min_value, fp_data[i]);
    //        _max_value = max(_max_value, fp_data[i]);
    //    }

    //    if (abs(_min_value) > abs(_max_value)) {
    //        _max_value = abs(_min_value);
    //    }
    //    else {
    //        _min_value = -abs(_max_value);
    //    }

    //    timer.stop();
    //    out() << "etermining floating point value range"
    //          << " (elapsed time: " << std::fixed << std::setprecision(3)
    //          << time::to_seconds(timer.get_time()) << "s)" << log::end;

    //}
    //out() << "min_value: " << _min_value << ", max_value: " << _max_value << log::end;

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
    texture_3d_ptr new_volume_tex = in_device.create_texture_3d(data_dimensions, data_format, mip_count, data_format, mip_init_data);
    timer.stop();
    out() << "allocating texture storage done."
          << " (elapsed time: " << std::fixed << std::setprecision(3)
          << time::to_seconds(timer.get_time()) << "s)" << log::end;

    std::for_each(mip_data.begin() + 1, mip_data.end(), [&mip_init_data](uint8* v) {delete [] v; });

    out() << log::outdent;

    return new_volume_tex;
}

scm::math::vec3ui
volume_loader::read_dimensions(const std::string&  in_image_path)
{
	using namespace scm::gl;
	using namespace scm::math;
	using namespace boost::filesystem;

	path                    file_path(in_image_path);
	std::string             file_name = file_path.filename().string();
	std::string             file_extension = file_path.extension().string();

	boost::algorithm::to_lower(file_extension);

	vec3ui      data_dimensions = vec3ui(0u);
	data_format data_format = FORMAT_NULL;
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
			<< "volume_data::load_volume(): unable to open file ('" << in_image_path << "')." << log::end;
		return scm::math::vec3ui::zero();
	}

	if (!(*vol_reader)) {
		err() << log::error
			<< "volume_data::load_volume(): unable to open file ('" << in_image_path << "')." << log::end;
		return scm::math::vec3ui::zero();
	}
	//out() << "source data dimensions: " << vol_reader->dimensions() << log::end;

	data_dimensions = vol_reader->dimensions();
	data_format = vol_reader->format();

	return data_dimensions;
}

} // namespace gl
} // namespace scm

