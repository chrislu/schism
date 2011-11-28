
#include "output_device_mode.h"

namespace gl
{
    output_device_mode NULL_DEVICE_MODE;

    output_device_mode::output_device_mode()
            : _bits_per_pixel(0),
              _width(0),
              _height(0),
              _refresh_rate(0)
    {
    }

    output_device_mode::output_device_mode(unsigned bpp,
                           unsigned w,
                           unsigned h,
                           unsigned ref)
                           : _bits_per_pixel(bpp),
                             _width(w),
                             _height(h),
                             _refresh_rate(ref)
    {
    }

    output_device_mode::output_device_mode(const output_device_mode& om)
    {
        _bits_per_pixel = om._bits_per_pixel;
        _width          = om._width;
        _height         = om._height;
        _refresh_rate   = om._refresh_rate;
    }

    bool output_device_mode::operator==(const gl::output_device_mode& om) const
    {
        bool tmp_ret = true;

        tmp_ret = tmp_ret && (_bits_per_pixel == om._bits_per_pixel);
        tmp_ret = tmp_ret && (_width          == om._width);
        tmp_ret = tmp_ret && (_height         == om._height);
        tmp_ret = tmp_ret && (_refresh_rate   == om._refresh_rate);

        return (tmp_ret);
    }

    bool output_device_mode::operator!=(const gl::output_device_mode& om) const
    {
       return (!(*this == om));
    }

} // namespace gl;