
#include "space_navigator_device.h"

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

#endif SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <iostream>

#include <scm/log.h>
#include <scm/gl/math/math.h>

[module(type=dll, name = "scm_input_sn")]
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

    void                update();

private:
    CComPtr<ISensor>    _3d_sensor;
    CComPtr<IKeyboard>  _3d_keyboard;

    scm::inp::space_navigator*const _device;
};

space_navigator_impl::space_navigator_impl(scm::inp::space_navigator*const d)
  : _device(d)
{
    HRESULT hr=::CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (!SUCCEEDED(hr)) {
        CString strError;
        strError.FormatMessage (_T("Error 0x%x"), hr);
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "space_navigator_impl::space_navigator_impl()"
                   << "CoInitializeEx failed: " << strError << std::endl;
    }

    CComPtr<IUnknown> _3DxDevice;

    hr = _3DxDevice.CoCreateInstance(__uuidof(Device));
    if (!SUCCEEDED(hr)) {
        CString strError;
        strError.FormatMessage (_T("Error 0x%x"), hr);
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "space_navigator_impl::space_navigator_impl()"
                   << "_3DxDevice.CoCreateInstance failed: " << strError << std::endl;
    }
    else {
        CComPtr<ISimpleDevice> _3DxSimpleDevice;

        hr = _3DxDevice.QueryInterface(&_3DxSimpleDevice);
        if (!SUCCEEDED(hr)) {
            CString strError;
            strError.FormatMessage (_T("Error 0x%x"), hr);
            scm::err() << scm::log_level(scm::logging::ll_error)
                       << "space_navigator_impl::space_navigator_impl()"
                       << "_3DxDevice.QueryInterface failed: " << strError << std::endl;
        }
        else {
            // Get the interfaces to the sensor and the keyboard;
            hr = _3DxSimpleDevice->get_Sensor(&_3d_sensor);
            hr = _3DxSimpleDevice->get_Keyboard(&_3d_keyboard);

            // Associate a configuration with this device
            //_3DxSimpleDevice->LoadPreferences(L"scm_input_sn");

            hr = __hook(&_ISensorEvents::SensorInput, _3d_sensor, 
                        &space_navigator_impl::on_sensor_input);

            hr = __hook(&_IKeyboardEvents::KeyDown, _3d_keyboard, 
                        &space_navigator_impl::on_key_down);

            hr = __hook(&_IKeyboardEvents::KeyUp, _3d_keyboard, 
                        &space_navigator_impl::on_key_up);

            // Connect to the driver
            hr = _3DxSimpleDevice->Connect();
            if (!SUCCEEDED(hr)) {
                CString strError;
                strError.FormatMessage (_T("Error 0x%x"), hr);
                scm::err() << scm::log_level(scm::logging::ll_error)
                           << "space_navigator_impl::space_navigator_impl()"
                           << "_3DxSimpleDevice.Connect failed: " << strError << std::endl;
            }
            else {
                scm::out() << "space_navigator_impl::space_navigator_impl(): successfully connected to _3DxSimpleDevice driver" << std::endl;
            }
        }
        //_3DxDevice.Release();
    }
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

    _3d_sensor->get_Rotation(&rotation);
    _3d_sensor->get_Translation(&translation);
    rotation->get_Angle(&rotation_angle);
    translation->get_Length(&translation_length);


    if (   rotation_angle > 0.0
        || translation_length > 0.0) {
        double time_factor = 1.0;

    std::cout << "update impl" << std::endl;
        DWORD time_stamp = ::GetTickCount();
        if (last_time_stamp) {
            double  period;
            _3d_sensor->get_Period(&period);
            time_factor = (double)(time_stamp - last_time_stamp) / period;
        }
        last_time_stamp = time_stamp;


        // translation
        vec3d  trans_vec;
        translation->get_X(&trans_vec.x);
        translation->get_Y(&trans_vec.y);
        translation->get_Z(&trans_vec.z);

        trans_vec *= vec3d(_device->_translation_sensitivity) * time_factor;
        translate(_device->_translation, math::vec3f(trans_vec));


        // rotation
        vec3d  rot_axis;
        rotation->get_X(&rot_axis.x);
        rotation->get_Y(&rot_axis.y);
        rotation->get_Z(&rot_axis.z);

        rotation_angle *= time_factor;
        rotate(_device->_rotation, (float)rotation_angle, math::vec3f(rot_axis));

        std::cout << _device->_rotation << std::endl;
    }
    rotation.Release();
    translation.Release();
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
