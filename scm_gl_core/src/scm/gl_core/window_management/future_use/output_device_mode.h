
#ifndef OUTPUT_DEVICE_MODE_H_INCLUDED
#define OUTPUT_DEVICE_MODE_H_INCLUDED

#include <defines_clr.h>

#include <vector>

#pragma warning (disable : 4561)
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>

namespace gl
{
    CLR_PUBLIC class output_device_mode
    {
    public:
        output_device_mode();
        output_device_mode(unsigned bpp,
                           unsigned w,
                           unsigned h,
                           unsigned ref);
        output_device_mode(const output_device_mode& om);

        bool operator==(const gl::output_device_mode& om) const;
        bool operator!=(const gl::output_device_mode& om) const;

        unsigned        _bits_per_pixel;
        unsigned        _width;
        unsigned        _height;
        unsigned        _refresh_rate;

    private:
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & BOOST_SERIALIZATION_NVP(_bits_per_pixel);
            ar & BOOST_SERIALIZATION_NVP(_width);
            ar & BOOST_SERIALIZATION_NVP(_height);
            ar & BOOST_SERIALIZATION_NVP(_refresh_rate);
        }
        friend class boost::serialization::access;

    };

    extern output_device_mode NULL_DEVICE_MODE;

} // namespace gl


#endif // OUTPUT_DEVICE_MODE_H_INCLUDED