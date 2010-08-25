
#include "output_device_win32.h"

#include <windows.h>

namespace gl
{
    output_device_win32::output_device_win32()
        : output_device()
    {
    }

    output_device_win32::~output_device_win32()
    {
        this->shutdown();
    }

    bool output_device_win32::initialize()
    {
        DEVMODE         dev_mode;
        DISPLAY_DEVICE  dev_info;

        dev_info.cb = sizeof(DISPLAY_DEVICE);

        if (!EnumDisplayDevices(NULL, 0, &dev_info, NULL)) {
            return (false);
        }

        // retrieve current display settings of the device the thread/app is running on
        if (!EnumDisplaySettings(dev_info.DeviceName, ENUM_CURRENT_SETTINGS, &dev_mode)) {
            return (false);
        }

        this->_device_mode._bits_per_pixel  = dev_mode.dmBitsPerPel;
        this->_device_mode._width           = dev_mode.dmPelsWidth;
        this->_device_mode._height          = dev_mode.dmPelsHeight;
        this->_device_mode._refresh_rate    = dev_mode.dmDisplayFrequency;
        this->_device_name                  = std::string((const char*)(dev_info.DeviceName));
        this->_device_string                = std::string((const char*)(dev_info.DeviceString));
        this->_device_id                    = std::string((const char*)(dev_info.DeviceID));

        this->_current_device_mode = this->_device_mode;

        return (true);
    }

    bool output_device_win32::switch_mode(const gl::output_device_mode& mode)
    {
        if (mode == gl::NULL_DEVICE_MODE) {
            if (ChangeDisplaySettingsEx(this->_device_name.c_str(), NULL, NULL, 0, NULL) != DISP_CHANGE_SUCCESSFUL) {
                return (false);
            }

            this->_current_device_mode = this->_device_mode;
        }
        else {
            DEVMODE dev_mode;
	        memset(&dev_mode, 0, sizeof(DEVMODE));
	        dev_mode.dmSize             = sizeof(DEVMODE);
	        dev_mode.dmFields           = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;
            dev_mode.dmBitsPerPel       = mode._bits_per_pixel;
            dev_mode.dmPelsWidth        = mode._width;
            dev_mode.dmPelsHeight       = mode._height;
            dev_mode.dmDisplayFrequency = mode._refresh_rate;

            if (ChangeDisplaySettingsEx(this->_device_name.c_str(), &dev_mode, NULL, CDS_FULLSCREEN, NULL) != DISP_CHANGE_SUCCESSFUL) {
                return (false);
            }

            this->_current_device_mode = mode;
        }

        return (true);
    }

    void output_device_win32::shutdown()
    {
        this->switch_mode(gl::NULL_DEVICE_MODE);
    }

} // namespace gl
