
#ifndef IMAGE_LOADER_H_INCLUDED
#define IMAGE_LOADER_H_INCLUDED

#include <string>

struct image
{
    image() : _id(0), _width(0), _height(0), _depth(0) {}

    unsigned        _id;

    unsigned        _width;
    unsigned        _height;
    unsigned        _depth;

    unsigned        _image_format;
    unsigned        _image_type;

    unsigned        _bpp;

};

bool init_image_loader();

void save_image(unsigned w, unsigned h, void* data, const std::string& name);
bool open_image(const std::string& file, image& img);
void close_image(image& img);

bool load_2d_texture(unsigned& tex_id, const image& img, bool gen_mip_maps = false);

#endif // IMAGE_LOADER_H_INCLUDED



