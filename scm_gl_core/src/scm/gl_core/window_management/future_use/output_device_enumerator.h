
#ifndef OUTPUT_DEVICE_ENUMERATOR_H_INCLUDED
#define OUTPUT_DEVICE_ENUMERATOR_H_INCLUDED

#include <defines_clr.h>

#include <string>
#include <vector>

#include <ogl/render_context/output_device_descriptor.h>

namespace gl
{
    CLR_PUBLIC class output_device_enumerator
    {
    public:
        typedef std::vector<std::string>    string_vec_t;
 
        output_device_enumerator();
        virtual ~output_device_enumerator();

        virtual bool                    enumerate_devices(std::vector<gl::output_device_descriptor>& devices) = 0;
        virtual bool                    enumerate_device(gl::output_device_descriptor& desc, unsigned device = 0) = 0;
        virtual bool                    enumerate_device(gl::output_device_descriptor& desc, const std::string& name) = 0;

        const string_vec_t&             get_feedback_messages() const { return (_issues); }

    protected:
        string_vec_t                    _issues;
    private:
    };
} // namespace gl


#endif // OUTPUT_DEVICE_ENUMERATOR_H_INCLUDED