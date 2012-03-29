
#include "height_field_data.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <stdexcept>
#include <limits>

#include <boost/assign/list_of.hpp>

#include <scm/log.h>
#include <scm/core/log/logger_state.h>

#include <scm/data/analysis/transfer_function/build_lookup_table.h>

#include <scm/gl_core/math.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/imaging/texture_image_data.h>
#include <scm/gl_util/imaging/texture_loader.h>
#include <scm/gl_util/primitives/box.h>

#include <renderers/patch_grid_mesh.h>

namespace scm {
namespace {

float
image_sample(const gl::texture_image_data_ptr& image_data,
             const math::vec2i&                sample_pos)
{
    using namespace scm::gl;
    using namespace scm::math;

    const vec2ui image_size  = image_data->mip_level(0).size();
    const vec2i  clamped_pos = clamp(sample_pos, vec2i(0, 0), vec2i(image_size) - vec2i(1, 1));

    if (image_data->format() == FORMAT_R_16) {
        const unsigned short*const data_ptr = reinterpret_cast<const unsigned short*const>(image_data->mip_level(0).data().get());
        const unsigned short       max_val  = (std::numeric_limits<unsigned short>::max)();
        const unsigned short       value    = data_ptr[clamped_pos.y * image_size.x + clamped_pos.x];

        return (static_cast<float>(value) / max_val);
    }
    else if (image_data->format() == FORMAT_R_8) {
        const unsigned char*const data_ptr = reinterpret_cast<const unsigned char*const>(image_data->mip_level(0).data().get());
        const unsigned char       max_val  = (std::numeric_limits<unsigned char>::max)();
        const unsigned char       value    = data_ptr[clamped_pos.y * image_size.x + clamped_pos.x];

        return (static_cast<float>(value) / max_val);
    }
    else if (image_data->format() == FORMAT_R_32F) {
        const float*const data_ptr = reinterpret_cast<const float*const>(image_data->mip_level(0).data().get());
        const float       value    = data_ptr[clamped_pos.y * image_size.x + clamped_pos.x];

        return (value);
    }

    return (0.0f);
}

struct null_deleter
{
    void operator()(void const *) const {
    }
}; // null_deleter

} // namespace

namespace data {

height_field_data::height_field_data(const gl::render_device_ptr& device,
                                     const std::string&           file_name,
                                     const math::vec3f&           height_field_extends)
  : _extends(height_field_extends)
  , _bbox(math::vec3f(0.0f), height_field_extends)
  , _transform(math::mat4f::identity())
{
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    const vec2ui            patch_size = vec2ui(64u);

    log::logger_format_saver save_indent(out().associated_logger());

    out() << log::info << "height_field_data::height_field_data()" << log::indent;
    out() << log::info << "loading height map image ('" << file_name << "')." << log::end;

    texture_loader          tex_loader;
    texture_image_data_ptr  image_data = tex_loader.load_image_data(file_name);

    if (!image_data) {
        throw (std::runtime_error("height_field_data::height_field_data(): error loading height field image data from file: " + file_name));
    }


    out() << log::info << "opened height field "
          << "(image size: " << image_data->mip_level(0).size()
          << ", patch size: " << patch_size
          << ", format: " << format_string(image_data->format()) << ")." << log::end;

    //out() << log::info << "padding height map to multiple of patch size "
    //      << "(image size: " << image_data->size()
    //      << ", patch size: " << patch_size
    //      << ", format: " << format_string(image_data->format()) << ")." << log::end;

    //image_data  = pad_to_patch_size(image_data, patch_size - 1);

    out() << log::info << "generating density map." << log::end;

    texture_image_data_ptr  density_data = generate_density_data(image_data, patch_size, height_field_extends);

    if (!density_data) {
        throw (std::runtime_error("height_field_data::height_field_data(): error generating density data."));
    }

    out() << log::info << "generating textures." << log::end;

    _height_map = device->create_texture_2d(image_data->mip_level(0).size(), image_data->format(), 1, 1, 1,
                                            image_data->format(), list_of(image_data->mip_level(0).data().get()));
    device->main_context()->generate_mipmaps(_height_map);

    _density_map = device->create_texture_2d(density_data->mip_level(0).size(), density_data->format(), 1, 1, 1,
                                             density_data->format(), list_of(density_data->mip_level(0).data().get()));

    if (!_height_map) {
        throw (std::runtime_error("height_field_data::height_field_data(): error loading height field texture."));
    }
    if (!_density_map) {
        throw (std::runtime_error("height_field_data::height_field_data(): error loading density texture"));
    }

    const vec2ui texture_size    = _height_map->descriptor()._size;
    const vec2ui patch_grid_res  = vec2ui(ceil(vec2f(texture_size) / vec2f(patch_size)));
    const vec2f  patch_grid_size = vec2f(patch_grid_res * patch_size) / vec2f(texture_size);
    const vec2f  patch_extends   = vec3f(patch_grid_size, height_field_extends.z);

    _transform = make_translation(-_extends / 2.0f);

    out() << log::info << "generation edge density buffers." << log::end;
    render_context_ptr ctx = device->main_context();
    { // tri edge density buffer
        _triangle_edge_density_buffer = device->create_texture_buffer(FORMAT_RGBA_32F, USAGE_STATIC_DRAW,
                                                                      patch_grid_res.x * patch_grid_res.y * 2 * 2 * sizeof(vec4f));// two triangles per grid cell, outer and inner tessellation factors
        if (!_triangle_edge_density_buffer) {
            throw (std::runtime_error("height_field_data::height_field_data(): error creating triangle edge density buffer."));
        }
        vec4f* data = static_cast<vec4f*>(ctx->map_buffer(_triangle_edge_density_buffer->descriptor()._buffer,
                                                          ACCESS_WRITE_INVALIDATE_BUFFER));

        if (!data) {
            throw (std::runtime_error("height_field_data::height_field_data(): unable map triangle edge density buffer."));
        }
        for (unsigned y = 0; y < patch_grid_res.y; ++y) {
            for (unsigned x = 0; x < patch_grid_res.x; ++x) {
                data[4 * (y * patch_grid_res.x + x) + 0] = vec4f(1.0f); // tri 0 outer
                data[4 * (y * patch_grid_res.x + x) + 1] = vec4f(1.0f); // tri 0 inner
                data[4 * (y * patch_grid_res.x + x) + 2] = vec4f(1.0f); // tri 1 outer
                data[4 * (y * patch_grid_res.x + x) + 3] = vec4f(1.0f); // tri 1 inner
            }
        }
        ctx->unmap_buffer(_triangle_edge_density_buffer->descriptor()._buffer);
    }

    out() << log::info << "generation color lookup texture." << log::end;
    _color_transfer.clear();

    _color_transfer.add_stop(0.0f,  vec3f(0.0f, 0.0f, 1.0f));
    _color_transfer.add_stop(0.25f, vec3f(0.0f, 1.0f, 1.0f));
    _color_transfer.add_stop(0.50f, vec3f(0.0f, 1.0f, 0.0f));
    _color_transfer.add_stop(0.75f, vec3f(1.0f, 1.0f, 0.0f));
    _color_transfer.add_stop(1.0f,  vec3f(1.0f, 0.0f, 0.0f));

    _color_map = create_color_map(*device, 512, _color_transfer);
    if (!_color_map) {
        throw std::runtime_error("height_field_data::height_field_data(): error creating color lookup map.");
    }

    out() << log::info << "generating patch and bbox geometries." << log::end;

    try {
        _bbox_geometry = make_shared<box_geometry>(device, _bbox.min_vertex(), _bbox.max_vertex());
        _patch_mesh    = make_shared<patch_grid_mesh>(device, texture_size, patch_size, vec2f(height_field_extends));
    }
    catch (const std::exception& e) {
        throw (std::runtime_error(std::string("height_field_data::height_field_data(): error creating geometry objects: ") + e.what()));
    }
}

height_field_data::~height_field_data()
{
    _bbox_geometry.reset();
    _height_map.reset();
    _density_map.reset();
    _quad_edge_density_buffer.reset();
    _triangle_edge_density_buffer.reset();
    _patch_mesh.reset();
}

const math::vec3f&
height_field_data::extends() const
{
    return (_extends);
}

const math::mat4f&
height_field_data::transform() const
{
    return (_transform);
}

void
height_field_data::transform(const math::mat4f& m)
{
    _transform = m;
}

const gl::box&
height_field_data::bbox() const
{
    return (_bbox);
}

const gl::box_geometry_ptr&
height_field_data::bbox_geometry() const
{
    return (_bbox_geometry);
}

const gl::texture_2d_ptr&
height_field_data::height_map() const
{
    return (_height_map);
}

const gl::texture_2d_ptr&
height_field_data::density_map() const
{
    return (_density_map);
}

const gl::texture_1d_ptr&
height_field_data::color_map() const
{
    return (_color_map);
}

const gl::texture_buffer_ptr&
height_field_data::quad_edge_density_buffer() const
{
    return (_quad_edge_density_buffer);
}

const gl::texture_buffer_ptr&
height_field_data::triangle_edge_density_buffer() const
{
    return (_triangle_edge_density_buffer);
}

const data::patch_grid_mesh_ptr&
height_field_data::patch_mesh() const
{
    return (_patch_mesh);
}

gl::texture_image_data_ptr
height_field_data::generate_density_data(const gl::texture_image_data_ptr& src_image,
                                         const math::vec2ui&               patch_size,
                                         const math::vec3f&                height_field_extends) const
{
    using namespace scm::gl;
    using namespace scm::math;

    out() << log::info << "height_field_data::generate_density_data()" << log::indent;

    scm::size_t         image_data_size = static_cast<size_t>(src_image->mip_level(0).size().x) * src_image->mip_level(0).size().y * sizeof(float);
    shared_array<uint8> image_gradients_array(new uint8[image_data_size]);
    float*              image_gradients = reinterpret_cast<float*>(image_gradients_array.get());

    vec2ui grid_resolution = vec2ui(floor(vec2f(vec2ui(src_image->mip_level(0).size())) / vec2f(patch_size - 1)));
    vec2f  os_patch_size   = vec2f(height_field_extends) / grid_resolution;
    float  max_os_patch_dim = max(os_patch_size.x, os_patch_size.y);

    out() << "grid_resolution " << grid_resolution << log::end;
    out() << "os_patch_size " << os_patch_size << log::end;
#if 0

    vec3f delta = height_field_extends / vec3f(static_cast<float>(src_image->size().x),
                                               static_cast<float>(src_image->size().y),
                                               1.0f);//pow(2.0f, 8 * size_of_channel(src_image->format())));

    float height_scale = 1.0f / pow(2.0f, 8 * size_of_channel(src_image->format()));

    for (int y = 0; y < static_cast<int>(src_image->size().y); ++y) {
        for (int x = 0; x < static_cast<int>(src_image->size().x); ++x) {
            //float c  = image_sample(src_image, vec2i(x,     y   ));
            float l  = image_sample(src_image, vec2i(x - 1, y   ));
            float r  = image_sample(src_image, vec2i(x + 1, y   ));
            float t  = image_sample(src_image, vec2i(x,     y + 1));
            float b  = image_sample(src_image, vec2i(x,     y - 1));
            //float ul = image_sample(src_image, vec2i(x - 1, y + 1));
            //float ll = image_sample(src_image, vec2i(x - 1, y - 1));
            //float ur = image_sample(src_image, vec2i(x + 1, y + 1));
            //float lr = image_sample(src_image, vec2i(x + 1, y - 1));

            //float dx = (r - l) * 0.5f;
            //float dy = (t - b) * 0.5f;
            //image_gradients[y * src_image->size().x + x] = clamp(sqrt(dx*dx + dy*dy), 0.0f, 1.0f);

            float dx = (r - l) * height_field_extends.z;
            float dy = (t - b) * height_field_extends.z;
            image_gradients[y * src_image->size().x + x] = clamp(sqrt(dx*dx + dy*dy), 0.0f, 1.0f) / max_os_patch_dim;

            //vec3f dx = vec3f(delta.x, 0.0f, (r - l) * delta.z);
            //vec3f dy = vec3f(0.0f, delta.y, (t - b) * delta.z);

            //image_gradients[y * src_image->size().x + x] = length(cross(dx, dy));
        }
    }
    texture_image_data_ptr image_gradient_data(new texture_image_data(src_image->size(), FORMAT_R_32F, image_gradients_array));

    vec2ui              density_image_size = vec2ui(floor(vec2f(src_image->size()) / (patch_size - 1)));
    shared_array<uint8> density_array(new uint8[density_image_size.x * density_image_size.y * sizeof(float)]);
    float*              density_image = reinterpret_cast<float*>(density_array.get());

    out() << log::info << "density image size: " << density_image_size << "." << log::end;

    for (int y = 0; y < static_cast<int>(density_image_size.y); ++y) {
        for (int x = 0; x < static_cast<int>(density_image_size.x); ++x) {
            float max_grad = -(std::numeric_limits<float>::max)();

            for (int iy = 0; iy < static_cast<int>(patch_size.y); ++iy) {
                for (int ix = 0; ix < static_cast<int>(patch_size.x); ++ix) {
                    float v = image_sample(image_gradient_data, vec2i(x * (patch_size.x - 1) + ix,
                                                                      y * (patch_size.y - 1) + iy));
                    max_grad = max(max_grad, v);
                }
            }

            density_image[y * density_image_size.x + x] = max_grad;
        }
    }

    texture_image_data_ptr density_data(new texture_image_data(density_image_size, FORMAT_R_32F, density_array));
#else
    texture_image_data::level_vector img_grad_mip_data;
    img_grad_mip_data.push_back(texture_image_data::level(src_image->mip_level(0).size(), image_gradients_array));
    texture_image_data_ptr image_gradient_data(new texture_image_data(texture_image_data::ORIGIN_LOWER_LEFT, FORMAT_R_32F, img_grad_mip_data));

    vec2ui              density_image_size = vec2ui(floor(vec2f(vec2ui(src_image->mip_level(0).size())) / (patch_size - 1)));
    shared_array<uint8> density_array(new uint8[density_image_size.x * density_image_size.y * sizeof(float)]);
    float*              density_image = reinterpret_cast<float*>(density_array.get());

    out() << log::info << "density image size: " << density_image_size << "." << log::end;

    for (int y = 0; y < static_cast<int>(density_image_size.y); ++y) {
        for (int x = 0; x < static_cast<int>(density_image_size.x); ++x) {
            float max_grad = -(std::numeric_limits<float>::max)();

            float min_val = (std::numeric_limits<float>::max)();
            float max_val = -(std::numeric_limits<float>::max)();

            for (int iy = 0; iy < static_cast<int>(patch_size.y); ++iy) {
                for (int ix = 0; ix < static_cast<int>(patch_size.x); ++ix) {

                    float v  = image_sample(src_image, vec2i(x * (patch_size.x - 1) + ix, y * (patch_size.y - 1) + iy));

                    min_val = min(min_val, v);
                    max_val = max(max_val, v);
                }
            }

            density_image[y * density_image_size.x + x] = (max_val - min_val) * height_field_extends.z / max_os_patch_dim;
        }
    }

    texture_image_data::level_vector img_dens_mip_data;
    img_dens_mip_data.push_back(texture_image_data::level(density_image_size, density_array));
    texture_image_data_ptr density_data(new texture_image_data(texture_image_data::ORIGIN_LOWER_LEFT, FORMAT_R_32F, img_dens_mip_data));
#endif
    return (density_data);
}

gl::texture_image_data_ptr
height_field_data::pad_to_patch_size(const gl::texture_image_data_ptr& src_image,
                                     const math::vec2ui&               patch_size) const
{
    using namespace scm::gl;
    using namespace scm::math;

    out() << log::info << "height_field_data::pad_to_patch_size()" << log::indent;

    vec2ui              padded_image_size      = vec2ui(ceil(vec2f(vec2ui(src_image->mip_level(0).size())) / patch_size)) * patch_size;
    scm::size_t         padded_image_data_size = static_cast<size_t>(padded_image_size.x) * padded_image_size.y * size_of_format(src_image->format());
    shared_array<uint8> padded_image_array(new uint8[padded_image_data_size]);

    out() << log::info << "padded image size: " << padded_image_size << "." << log::end;

    if (src_image->format() == FORMAT_R_8) {
        unsigned char* padded_image = reinterpret_cast<unsigned char*>(padded_image_array.get());
        unsigned char* source_image = reinterpret_cast<unsigned char*>(src_image->mip_level(0).data().get());

        // clear data array to 0
        std::fill(padded_image, padded_image + (static_cast<size_t>(padded_image_size.x) * padded_image_size.y), 0);

        for (int y = 0; y < static_cast<int>(src_image->mip_level(0).size().y); ++y) {
            memcpy(padded_image + y * padded_image_size.x,
                   source_image + y * src_image->mip_level(0).size().x,
                   src_image->mip_level(0).size().x * size_of_format(src_image->format()));
        }
    }
    else if (src_image->format() == FORMAT_R_16) {
        unsigned short* padded_image = reinterpret_cast<unsigned short*>(padded_image_array.get());
        unsigned short* source_image = reinterpret_cast<unsigned short*>(src_image->mip_level(0).data().get());

        // clear data array to 0
        std::fill(padded_image, padded_image + (static_cast<size_t>(padded_image_size.x) * padded_image_size.y), 0);

        for (int y = 0; y < static_cast<int>(src_image->mip_level(0).size().y); ++y) {
            memcpy(padded_image + y * padded_image_size.x,
                   source_image + y * src_image->mip_level(0).size().x,
                   src_image->mip_level(0).size().x * size_of_format(src_image->format()));
        }
    }
    else if (src_image->format() == FORMAT_R_32F) {
        float*              padded_image           = reinterpret_cast<float*>(padded_image_array.get());
        float*              source_image           = reinterpret_cast<float*>(src_image->mip_level(0).data().get());

        // clear data array to 0.0f
        std::fill(padded_image, padded_image + (static_cast<size_t>(padded_image_size.x) * padded_image_size.y), 0.0f);

        for (int y = 0; y < static_cast<int>(src_image->mip_level(0).size().y); ++y) {
            memcpy(padded_image + y * padded_image_size.x,
                   source_image + y * src_image->mip_level(0).size().x,
                   src_image->mip_level(0).size().x * size_of_format(src_image->format()));
        }
    }

    texture_image_data::level_vector img_pad_mip_data;
    img_pad_mip_data.push_back(texture_image_data::level(padded_image_size, padded_image_array));
    texture_image_data_ptr padded_data(new texture_image_data(texture_image_data::ORIGIN_LOWER_LEFT, src_image->format(), img_pad_mip_data));

    return (padded_data);
}

gl::texture_1d_ptr
height_field_data::create_color_map(gl::render_device& in_device,
                                    unsigned in_size,
                                    const color_transfer_type& in_color) const
{
    using namespace scm::gl;
    using namespace scm::math;

    log::logger_format_saver save_indent(out().associated_logger());
    out() << log::indent;

    scm::scoped_array<scm::math::vec3f>  color_lut;

    color_lut.reset(new vec3f[in_size]);

    if (   !scm::data::build_lookup_table(color_lut, in_color, in_size)) {
        scm::err() << "height_field_data::create_color_map(): error during lookuptable generation" << log::end;
        return (texture_1d_ptr());
    }
    scm::scoped_array<float> combined_lut;

    combined_lut.reset(new float[in_size * 4]);

    for (unsigned i = 0; i < in_size; ++i) {
        combined_lut[i*4   ] = color_lut[i].x;
        combined_lut[i*4 +1] = color_lut[i].y;
        combined_lut[i*4 +2] = color_lut[i].z;
        combined_lut[i*4 +3] = 1.0f;
    }

    std::vector<void*> in_data;
    in_data.push_back(combined_lut.get());

    texture_1d_ptr new_tex = in_device.create_texture_1d(in_size, FORMAT_RGBA_8, 1, 1, FORMAT_RGBA_32F, in_data);

    if (!new_tex) {
        scm::err() << "height_field_data::create_color_map(): error during color map texture generation." << log::end;
        return (texture_1d_ptr());
    }

    return (new_tex);
}

} // namespace data
} // namespace scm
