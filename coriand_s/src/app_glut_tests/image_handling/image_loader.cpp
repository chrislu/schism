
#include "image_loader.h"

#include <iostream>

#include <IL/il.h>

#include <scm/ogl.h>

bool init_image_loader()
{
    ilInit();

    return (true);
}

void save_image(unsigned w, unsigned h, void* data, const std::string& name)
{
    std::string file_name = name + std::string(".jpg");
    unsigned    image = 0;

    ilSetInteger(IL_JPG_QUALITY, 99);
    ilEnable(IL_ORIGIN_SET);
    ilOriginFunc(IL_ORIGIN_LOWER_LEFT);

    ilGenImages(1, &image);
    ilBindImage(image);
    ilTexImage(w, h, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, data);

    ilSaveImage(const_cast<char*>(file_name.c_str()));
    ilDeleteImages(1, &image);
}

bool open_image(const std::string& file, image& img)
{
    ilGenImages(1, &img._id);
    ilBindImage(img._id);

    ilEnable(IL_ORIGIN_SET);
    ilOriginFunc(IL_ORIGIN_LOWER_LEFT);

    if (!ilLoadImage((const ILstring)file.c_str()))
    {
        switch (ilGetError())
        {
        case IL_COULD_NOT_OPEN_FILE: std:: cout << "The file could not be opened. Either the file does not exist or is in use by another process." << std::endl;break;
        case IL_INVALID_EXTENSION:   std:: cout << "The file could not be loaded based on extension or header." << std::endl;break;
        case IL_INVALID_PARAM:       std:: cout << "FileName was not valid. It was most likely NULL." << std::endl;break;
        default:                     std:: cout << "unknown error occured during ilLoadImage" << std::endl;break;
        }

        return (false);
    }

    img._width                = ilGetInteger(IL_IMAGE_WIDTH);
    img._height               = ilGetInteger(IL_IMAGE_HEIGHT);
    img._depth                = ilGetInteger(IL_IMAGE_DEPTH);

    img._image_format         = ilGetInteger(IL_IMAGE_FORMAT);
    img._image_type           = ilGetInteger(IL_IMAGE_TYPE);

    img._bpp                  = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);

    ilBindImage(0);

    return (true);
}

void close_image(image& img)
{
    ilDeleteImages(1, &img._id);
}


bool load_2d_texture(unsigned& tex_id, const image& img, bool gen_mip_maps)
{
    if (tex_id == 0) {
        std::cout << "error: illegal texture id" << std::endl;
        return (false);
    }

    if (img._image_type != IL_UNSIGNED_BYTE) {
        std::cout << "error: unsupported image type" << std::endl;
        return (false);
    }

    GLenum source_type = GL_UNSIGNED_BYTE;
    GLenum internal_format;
    GLenum source_format;

    switch (img._image_format) {
        case IL_LUMINANCE:          internal_format = GL_LUMINANCE; source_format = GL_LUMINANCE; break;
        case IL_LUMINANCE_ALPHA:    internal_format = GL_LUMINANCE_ALPHA; source_format = GL_LUMINANCE_ALPHA; break;
        case IL_BGR:                internal_format = GL_RGB; source_format = GL_BGR; break;
        case IL_BGRA:               internal_format = GL_RGBA; source_format = GL_BGRA; break;
        case IL_RGB:                internal_format = GL_RGB; source_format = GL_RGB; break;
        case IL_RGBA:               internal_format = GL_RGBA; source_format = GL_RGBA; break;
        default: return (false);
    }

    ilBindImage(img._id);

    glBindTexture(GL_TEXTURE_2D, tex_id);
    glEnable(GL_TEXTURE_2D);

    if (gen_mip_maps) {
        gluBuild2DMipmaps(GL_TEXTURE_2D, internal_format, img._width, img._height, source_format, source_type, (void*)ilGetData());
        if (glGetError() != GL_NO_ERROR) {
            std::cout << "error: texture upload failed" << std::endl;
            return (false);
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    }
    else {
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, img._width, img._height, 0, source_format, source_type, (void*)ilGetData());
        if (glGetError() != GL_NO_ERROR) {
            std::cout << "error: texture upload failed" << std::endl;
            return (false);
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    ilBindImage(0);

    return (true);
}



