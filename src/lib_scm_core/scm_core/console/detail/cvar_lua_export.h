
#ifndef CVAR_LUA_EXPORT_H_INCLUDED
#define CVAR_LUA_EXPORT_H_INCLUDED

struct lua_State;

#include <scm_core/platform/platform.h>

namespace scm {
namespace con {
namespace detail {

void __scm_export cvar_export_lua_binding(lua_State* l_state);

} // namespace detail
} // namespace con
} // namespace scm

#endif // CVAR_LUA_EXPORT_H_INCLUDED
