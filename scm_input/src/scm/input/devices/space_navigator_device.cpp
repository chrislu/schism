
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "space_navigator_device.h"

#include <iostream>
#include <cassert>

#include <scm/log.h>
#include <scm/time.h>

#include <scm/gl_core/math.h>

#include <scm/input/config.h>

#ifdef SCM_INPUT_DEVICE_SPACE_NAVIGATOR

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

// com stuff for the driver callbacks
//#include <scm/core/platform/windows.h>

#define _ATL_ATTRIBUTES 1
#define _WIN32_DCOM

#ifndef WINVER			// Allow use of features specific to Windows XP or later.
#define WINVER 0x0501		// Change this to to target other versions of Windows.
#endif

#ifndef _WIN32_WINNT		// Allow use of features specific to WinXP or later.
#define _WIN32_WINNT 0x0501	// (was 0x0501) Change to target other versions of Windows
#endif						

#ifndef _WIN32_WINDOWS		// Allow use of features specific to Windows 98 or later.
#define _WIN32_WINDOWS 0x0410 // Change this to target Windows Me or later.
#endif

#ifndef _WIN32_IE			// Allow use of features specific to IE 6.0 or later.
#define _WIN32_IE 0x0600	// Change this to target other versions of IE.
#endif

#define _ATL_APARTMENT_THREADED
#define _ATL_NO_AUTOMATIC_NAMESPACE
#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS	// some CString constructors will be explicit
//#define _ATL_APARTMENT_THREADED
//#define _ATL_NO_AUTOMATIC_NAMESPACE
//
//#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS	// some CString constructors will be explicit

#include <atlbase.h>
#include <atlcom.h>
#include <atlwin.h>
#include <atltypes.h>
#include <atlctl.h>
#include <atlhost.h>
#include <atlstr.h>

using namespace ATL;

#import "progid:TDxInput.Device" embedded_idl no_namespace
// end com stuff

[module(type=dll, name = "scm_input_sn")]
class atl_scm_sn_module // The atl module class which the module attribute belongs to
{
};

[event_receiver(com)]
class space_navigator_impl
{
public:
    space_navigator_impl(scm::inp::space_navigator*const d = 0);
    virtual ~space_navigator_impl();

public:
    HRESULT             on_sensor_input();
    HRESULT             on_key_down(int k);
    HRESULT             on_key_up(int k);

    void                init_com();
    void                update();

private:
    CComPtr<ISensor>    _3d_sensor;
    CComPtr<IKeyboard>  _3d_keyboard;

    scm::inp::space_navigator*const _device;

    scm::time::high_res_timer       _timer;
};

space_navigator_impl::space_navigator_impl(scm::inp::space_navigator*const d)
  : _device(d)
{
    init_com();

    _timer.start();
    _timer.stop();

}

space_navigator_impl::~space_navigator_impl()
{
    HRESULT hr = E_FAIL;

    CComPtr<IDispatch> _3DxDevice;
    if (_3d_sensor) {
        hr = _3d_sensor->get_Device(&_3DxDevice);
    }
    else if (_3d_keyboard) {
        hr = _3d_keyboard->get_Device(&_3DxDevice);
    }

    if (SUCCEEDED(hr)) {
        CComPtr<ISimpleDevice> _3DxSimpleDevice;
        hr = _3DxDevice.QueryInterface(&_3DxSimpleDevice);
        if (SUCCEEDED(hr)) {
            _3DxSimpleDevice->Disconnect();
            _3DxSimpleDevice.Release();
        }
    }

    if (_3d_sensor) {
        // unhook (unadvise) the sensor event sink
        __unhook(&_ISensorEvents::SensorInput, _3d_sensor, 
                 &space_navigator_impl::on_sensor_input);
        _3d_sensor.Release();
    }

    if (_3d_keyboard) {
        __unhook(&_IKeyboardEvents::KeyDown, _3d_keyboard,
                 &space_navigator_impl::on_key_down);
        __unhook(&_IKeyboardEvents::KeyUp, _3d_keyboard,
                 &space_navigator_impl::on_key_up);
        _3d_keyboard.Release();
    }
}

void
space_navigator_impl::update()
{
    static DWORD last_time_stamp = 0;

    using namespace scm;
    using namespace scm::math;

    CComPtr<IAngleAxis> rotation;
    CComPtr<IVector3D>  translation;
    double              rotation_angle;
    double              translation_length;

    if (_3d_sensor) {
        _3d_sensor->get_Rotation(&rotation);
        _3d_sensor->get_Translation(&translation);
        rotation->get_Angle(&rotation_angle);
        translation->get_Length(&translation_length);

        _device->reset();

        if (   (rotation_angle     > 0.0)
            || (translation_length > 0.0))
        {

            _timer.stop();
            _timer.start();

            double time_factor = 1.0;
            double  period; // in millisec
            _3d_sensor->get_Period(&period);
            double frame_time = scm::time::to_milliseconds(_timer.get_time());
            // detect interaction pauses
            //if (frame_time > 250.0)
            {
                frame_time = period;
            }
            time_factor = frame_time / (period * 1000.0);
            //DWORD time_stamp = ::GetTickCount();
            //if (last_time_stamp) {
            //    double  period; // in millisec
            //    _3d_sensor->get_Period(&period);
            //    time_factor = (double)(time_stamp - last_time_stamp) / (1000.0 * period);
                //std::cout << period << " " << time_factor << std::endl;
            //}

            //DWORD time_stamp = ::GetTickCount(); 
            //if (last_time_stamp) { 
            //    double  period; 
            //    _3d_sensor->get_Period(&period); 
            //    time_factor = (double)(time_stamp - last_time_stamp) / (1000.0 * period); 
            //    //std::cout << period << " " << time_factor << std::endl; 
            //} 
            //last_time_stamp = time_stamp;


            // translation
            vec3d  trans_vec;
            translation->get_X(&trans_vec.x);
            translation->get_Y(&trans_vec.y);
            translation->get_Z(&trans_vec.z);
            
            //std::cout << trans_vec << " ";
            //std::cout << trans_vec / 40.0 << " ";
            trans_vec.x *= pow(trans_vec.x / 40.0, 2.0);
            trans_vec.y *= pow(trans_vec.y / 40.0, 2.0);
            trans_vec.z *= pow(trans_vec.z / 40.0, 2.0);
            //std::cout << trans_vec / 40.0 << std::endl;
            trans_vec   *= vec3d(_device->_translation_sensitivity) * time_factor;

            translate(_device->_translation, math::vec3f(trans_vec));


            // rotation
            vec3d  rot_axis;
            rotation->get_X(&rot_axis.x);
            rotation->get_Y(&rot_axis.y);
            rotation->get_Z(&rot_axis.z);

            //std::cout << std::fixed << std::setprecision(3)
            //          << scm::time::to_milliseconds(_timer.get_time()) << "\t"
            //          << trans_vec << "\t" << rot_axis << "\t" << rotation_angle << std::endl;

            //std::cout << rotation_angle << std::endl;
            rotation_angle *= pow(rotation_angle / 40.0, 2.0);

            rotation_angle *= time_factor;
            rotate(_device->_rotation, static_cast<float>(math::rad2deg(rotation_angle)), (math::vec3f(rot_axis)));
        }
        rotation.Release();
        translation.Release();
    }
}

void
space_navigator_impl::init_com()
{
    HRESULT hr = ::CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_SPEED_OVER_MEMORY);
    if (!SUCCEEDED(hr)) {
        CString strError;
        strError.FormatMessage (_T("Error 0x%x"), hr);
        scm::err() << scm::log::error
                   << "space_navigator_impl::space_navigator_impl()"
                   << "CoInitializeEx failed: " << strError << scm::log::end;
    }

    CComPtr<IUnknown> _3DxDevice;

    hr = _3DxDevice.CoCreateInstance(__uuidof(Device));
    if (!SUCCEEDED(hr)) {
        CString strError;
        strError.FormatMessage (_T("Error 0x%x"), hr);
        scm::err() << scm::log::error
                   << "space_navigator_impl::space_navigator_impl()"
                   << "_3DxDevice.CoCreateInstance failed: " << strError << scm::log::end;
    }
    else {
        CComPtr<ISimpleDevice> _3DxSimpleDevice;

        hr = _3DxDevice.QueryInterface(&_3DxSimpleDevice);
        if (!SUCCEEDED(hr)) {
            CString strError;
            strError.FormatMessage (_T("Error 0x%x"), hr);
            scm::err() << scm::log::error
                       << "space_navigator_impl::space_navigator_impl()"
                       << "_3DxDevice.QueryInterface failed: " << strError << scm::log::end;
        }
        else {
            // Get the interfaces to the sensor and the keyboard;
            hr = _3DxSimpleDevice->get_Sensor(&_3d_sensor);
            hr = _3DxSimpleDevice->get_Keyboard(&_3d_keyboard);

            // Associate a configuration with this device
            //_3DxSimpleDevice->LoadPreferences(L"scm_input_sn");

            //hr = __hook(&_ISensorEvents::SensorInput, _3d_sensor, 
            //            &space_navigator_impl::on_sensor_input);

            hr = __hook(&_IKeyboardEvents::KeyDown, _3d_keyboard, 
                        &space_navigator_impl::on_key_down);

            hr = __hook(&_IKeyboardEvents::KeyUp, _3d_keyboard, 
                        &space_navigator_impl::on_key_up);

            // Connect to the driver
            hr = _3DxSimpleDevice->Connect();
            if (!SUCCEEDED(hr)) {
                CString strError;
                strError.FormatMessage (_T("Error 0x%x"), hr);
                scm::err() << scm::log::error
                           << "space_navigator_impl::space_navigator_impl()"
                           << "_3DxSimpleDevice.Connect failed: " << strError << scm::log::end;
            }
            else {
                scm::out() << "space_navigator_impl::space_navigator_impl(): successfully connected to _3DxSimpleDevice driver" << scm::log::end;
            }
        }
        //_3DxDevice.Release();
    }
}

HRESULT
space_navigator_impl::on_sensor_input()
{
    if (!_device) {
        return (S_FALSE);
    }

    update();

    return (S_OK);
}

HRESULT
space_navigator_impl::on_key_down(int k)
{
    return (S_OK);
}

HRESULT
space_navigator_impl::on_key_up(int k)
{
    return (S_OK);
}


#else // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

class space_navigator_impl
{
public:
    space_navigator_impl(scm::inp::space_navigator*const d = 0) {};
    virtual ~space_navigator_impl() {};

public:
    void                update() {};

private:
};

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#else // SCM_INPUT_DEVICE_SPACE_NAVIGATOR

class space_navigator_impl
{
public:
    space_navigator_impl(scm::inp::space_navigator*const d = 0) {};
    virtual ~space_navigator_impl() {};

public:
    void                update() {};

private:
};

#endif // SCM_INPUT_DEVICE_SPACE_NAVIGATOR

namespace scm {
namespace inp {

namespace detail {


} // namespace detail

space_navigator::space_navigator()
{
    _rotation_sensitivity    = math::vec3f::one();
    _translation_sensitivity = math::vec3f::one();

    _rotation    = math::mat4f::identity();
    _translation = math::mat4f::identity();

    _device = make_shared<space_navigator_impl>(this);
}

space_navigator::~space_navigator()
{
    _device.reset();
}

void
space_navigator::update()
{
    //std::cout << "update dev" << std::endl;
    _device->update();
}

void
space_navigator::reset()
{
    _rotation    = math::mat4f::identity();
    _translation = math::mat4f::identity();
}

const math::mat4f&
space_navigator::rotation() const
{
    return (_rotation);
}

const math::mat4f&
space_navigator::translation() const
{
    return (_translation);
}

} // namespace inp
} // namespace scm

