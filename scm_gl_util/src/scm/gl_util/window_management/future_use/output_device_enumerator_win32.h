
#ifndef OUTPUT_DEVICE_ENUMERATOR_WIN32_H_INCLUDED
#define OUTPUT_DEVICE_ENUMERATOR_WIN32_H_INCLUDED

#include <defines_clr.h>

#include <ogl/render_context/output_device_enumerator.h>

#include <windows.h>

namespace gl
{
    CLR_PUBLIC class output_device_enumerator_win32 : public output_device_enumerator
    {
    public:
        output_device_enumerator_win32();
        virtual ~output_device_enumerator_win32();

        bool                    enumerate_devices(std::vector<gl::output_device_descriptor>& devices);
        bool                    enumerate_device(gl::output_device_descriptor& desc, unsigned device = 0);
        bool                    enumerate_device(gl::output_device_descriptor& desc, const std::string& name);
    protected:
    private:
    };
} // namespace gl


#endif // OUTPUT_DEVICE_ENUMERATOR_WIN32_H_INCLUDED