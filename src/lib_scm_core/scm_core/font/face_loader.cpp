
#include "face_loader.h"

#include <scm_core/font/face.h>

#include <scm_core/font/font_resource_manager.h>
#include <scm_core/resource/resource.h>

#include <scm_core/font/font_resource.h>



struct gl_face_descriptor : public scm::font::face_descriptor
{
    gl_face_descriptor() : _tex_id(0) {}
    gl_face_descriptor(const gl_face_descriptor& desc) : scm::font::face_descriptor(desc), _tex_id(desc._tex_id) {}

    std::string         _name;
    unsigned            _size;
    unsigned            _tex_id;
};

inline bool operator<(const gl_face_descriptor& lhs,
                      const gl_face_descriptor& rhs)
{
    return   ((lhs._name < rhs._name)
           || (lhs._size < rhs._size)
           || (lhs._tex_id < rhs._tex_id));
}

class gl_font_face : public scm::res::resource_type_base<gl_face_descriptor>
{
public:
    gl_font_face() {}
    virtual ~gl_font_face() {}

};

class gl_font_resource_manager : public scm::res::resource_manager<gl_font_face>
{
public:
    gl_font_resource_manager();
    virtual ~gl_font_resource_manager();

};

//typedef scm::res::resource_manager<gl_font_face>        gl_font_resource_manager;
typedef scm::res::resource<gl_font_face>    gl_font;

using namespace scm::font;

face_loader::face_loader()
    : _font_directory(std::string(SCM_RESSOURCE_DIR) + "fonts")
{
}

face_loader::face_loader(const std::string& font_dir)
    : _font_directory(font_dir)
{
}

face_loader::~face_loader()
{
}

bool face_loader::load(const std::string& file_name,
                       unsigned           size,
                       unsigned           disp_res)
{
    ::font a;
    ::font b;

    if (a) {
        a.get()._advance;
    }

    if (!a) {
        a.get()._advance;
    }

    gl_font c;
    gl_font d;

    std::swap(c,d);

    std::swap(a, b);

    //face*   new_face = new 



    return (false);
}