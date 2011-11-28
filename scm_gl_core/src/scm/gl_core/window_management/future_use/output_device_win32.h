
#ifndef OUTPUT_DEVICE_WIN32_H_INCLUDED
#define OUTPUT_DEVICE_WIN32_H_INCLUDED

#include <defines_clr.h>

#include <ogl/render_context/output_device.h>

namespace gl
{
    CLR_PUBLIC class output_device_win32 : public gl::output_device
    {
    public:
        output_device_win32();
        virtual ~output_device_win32();

        bool                    initialize();
        bool                    switch_mode(const gl::output_device_mode& mode);
        void                    shutdown();

    protected:

    private:
    }; // class output_device_win32

} // namespace gl

#endif // OUTPUT_DEVICE_WIN32_H_INCLUDED