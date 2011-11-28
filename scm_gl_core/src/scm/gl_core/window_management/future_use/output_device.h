
#ifndef OUTPUT_DEVICE_H_INCLUDED
#define OUTPUT_DEVICE_H_INCLUDED

#include <defines_clr.h>

#include <ogl/render_context/context_format.h>
#include <ogl/render_context/output_device_mode.h>

namespace gl
{
    CLR_PUBLIC class output_device
    {
    public:
        output_device();
        virtual ~output_device();

        virtual bool                    initialize() = 0;
        virtual bool                    switch_mode(const gl::output_device_mode& mode) = 0;
        virtual void                    shutdown() = 0;

        const std::string&              get_device_name() const     { return (this->_device_name); }
        const std::string&              get_device_string() const   { return (this->_device_string); }
        const std::string&              get_device_id() const       { return (this->_device_id); }
        const gl::output_device_mode&   get_device_mode() const     { return (this->_current_device_mode); }

    protected:
        gl::output_device_mode          _device_mode;
        gl::output_device_mode          _current_device_mode;

        std::string                     _device_name;
        std::string                     _device_string;
        std::string                     _device_id;


    private:
    }; // class output_device
} // namespace gl

#endif // OUTPUT_DEVICE_H_INCLUDED