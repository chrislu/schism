
#include <scm/core/utilities/luabind_warning_disable.h>

#include "cvar_lua_export.h"

#if 0

#include <string>

extern "C"
{
    #include <lua.h>
    #include <lualib.h>
    #include <lauxlib.h>
}

#include <luabind/luabind.hpp>
#include <luabind/operator.hpp>

#include <scm/core/console/cvar.h>

namespace scm {
namespace con {
namespace detail {

void cvar_export_lua_binding(lua_State* l_state)
{
    using namespace luabind;

    module(l_state)
    [
        class_<cvar>("cvar")
            .enum_("cvar_type")
            [
                value("CVAR_T_NUMBER", cvar::CVAR_T_NUMBER),
                value("CVAR_T_STRING", cvar::CVAR_T_STRING),
                value("CVAR_T_UNASSIGNED", cvar::CVAR_T_UNASSIGNED)
            ]
            .def(constructor<>())
            .def(constructor<const cvar&>())
            .def(tostring(const_self))

            // set
            .def("set_value",           (void(cvar::*)(cvar::number_t))&cvar::set_value)
            .def("set_value",           (void(cvar::*)(const std::string&))&cvar::set_value)

            // get
            .def("get_number_value",    (cvar::number_t(cvar::*)())&cvar::get_number_value)
            .def("get_string_value",    (const std::string&(cvar::*)())&cvar::get_string_value)

            // get_type
            .def("get_type",            (cvar::cvar_type(cvar::*)())&cvar::get_type)

            // equals
            .def("equals",              (bool(cvar::*)(const cvar&) const)&cvar::equals)

            // swap
            //.def("swap",                (void(cvar::*)(cvar&))&cvar::equals)

            // operators !!! attention only works completely with
            //           !!! equality operator lua power patch
            .def(const_self == other<cvar>())
            .def(const_self == cvar::number_t())
            .def(const_self == other<std::string>())

            .def(cvar::number_t() == const_self)
            .def(other<std::string>() == const_self)
    ];
}

} // namespace detail
} // namespace con
} // namespace scm

#endif

#include <scm/core/utilities/luabind_warning_enable.h>

