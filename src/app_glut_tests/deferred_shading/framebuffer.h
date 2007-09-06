
#ifndef GLUT_TESTS_DS_FRAMEBUFFER_H_INCLUDED
#define GLUT_TESTS_DS_FRAMEBUFFER_H_INCLUDED

namespace scm {

class ds_framebuffer
{
public:
    ds_framebuffer(unsigned /*width*/,
                   unsigned /*height*/);
    virtual ~ds_framebuffer();

    void            bind();
    void            unbind();

    unsigned        id() const          { return (_id); };
    unsigned        depth_id() const    { return (_depth_id); };
    unsigned        color_id() const    { return (_color_id); };
    unsigned        normal_id() const   { return (_normal_id); };

protected:

private:
    unsigned        _id;
    unsigned        _depth_id;
    unsigned        _color_id;
    unsigned        _normal_id;

    bool            init_textures(unsigned /*width*/,
                                  unsigned /*height*/);
    bool            init_fbo();
    void            cleanup();

}; // class ds_framebuffer

} // namespace scm

#endif // GLUT_TESTS_DS_FRAMEBUFFER_H_INCLUDED
