
#ifndef FRAMEBUFFER_OBJECT_H_INCLUDED
#define FRAMEBUFFER_OBJECT_H_INCLUDED

namespace gl
{
    class renderbuffer;

    class framebuffer_object
    {
    public:
        framebuffer_object();
        virtual ~framebuffer_object();

        bool    bind();

        bool    bind_to_draw();
        bool    bind_to_read();

        static void    unbind();

        static void    unbind_from_read();
        static void    unbind_from_draw();

    protected:

    private:
        framebuffer_object(const framebuffer_object&);
        const framebuffer_object& operator=(const framebuffer_object&);

    }; // class framebuffer_object


} // namespace gl

#endif // FRAMEBUFFER_OBJECT_H_INCLUDED
