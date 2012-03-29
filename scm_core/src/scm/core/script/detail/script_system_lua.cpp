
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <scm/core/utilities/luabind_warning_disable.h>

#include <boost/algorithm/string/trim.hpp>

#include "script_system_lua.h"

#include <scm/core/platform/platform.h>

#if 0

#include <scm/core/core.h>
#include <scm/core/console/console.h>

using namespace scm::core;
using namespace scm::detail;

extern "C"
{
    #include <lua.h>
    #include <lualib.h>
    #include <lauxlib.h>
}
#include <luabind/luabind.hpp>


namespace
{
    // stream reader for use in lua_load
    struct lua_istream_reader_data {
        lua_istream_reader_data(std::istream& str)
            : _stream(str) {}

        std::istream&       _stream;
    };

    static const char* lua_istream_reader(lua_State* l_state, void* data, size_t* size)
    {
        static lua_istream_reader_data* in_data;
        static std::string              cur_line;

        in_data = static_cast<lua_istream_reader_data*>(data);

        if(!in_data->_stream.eof()) {
            std::getline(in_data->_stream, cur_line);

            // remove all whitespace form beginning and end
            // counters line ending problems
            boost::algorithm::trim(cur_line);
            cur_line += "\n";

            *size   = cur_line.size();
            return (cur_line.c_str());
        }
        else {
            return (0);
        }
    }

    // string reader for use in lua_load
    struct lua_string_reader_data {
        lua_string_reader_data(const std::string& str)
            : _string(str), _done_reading(false) {}

        const std::string&  _string;
        bool                _done_reading;
    };

    static const char* lua_string_reader(lua_State* l_state, void* data, size_t* size)
    {
        static lua_string_reader_data*  in_data;

        in_data = static_cast<lua_string_reader_data*>(data);

        if (!in_data->_done_reading) {
            *size   = in_data->_string.size();
            return (in_data->_string.c_str());
        }
        else {
            return (0);
        }
    }

} // namespace

// script_system_lua implementation
script_system_lua::script_system_lua()
    : _l_incomplete_input_msg("near '<eof>'")
{
}

script_system_lua::~script_system_lua()
{
}
    
bool script_system_lua::initialize()
{
    _l_state = lua_open();

    if (_l_state == 0) {
        console.get() << "script_system_lua::initialize(): (error)"
                      << "unable to create lua_State (lua_open() returned NULL)" << std::endl;
        return (false);
    }

    luaopen_base(_l_state);
    luabind::open(_l_state);

    _initialized = true;

    return (true);
}

bool script_system_lua::shutdown()
{
    lua_close(_l_state);
    return (true);
}

script_result_t script_system_lua::process_script(std::istream& in_stream,
                                                  const std::string& input_source_name)
{
    lua_istream_reader_data     tmp_stream_data = lua_istream_reader_data(in_stream);

    return (int_process_script(lua_istream_reader,
                               &tmp_stream_data,
                               input_source_name));
}

script_result_t script_system_lua::process_script(const std::string& in_string,
                                                  const std::string& input_source_name)
{
    lua_string_reader_data      tmp_string_data = lua_string_reader_data(in_string);

    return (int_process_script(lua_string_reader,
                               &tmp_string_data,
                               input_source_name));
}

script_result_t script_system_lua::int_process_script(lua_Reader input_reader,
                                                      void* in_data,
                                                      const std::string& input_source_name)
{
    int             l_status    = 0;
    const char*     l_response;
    #ifdef SCM_DEBUG
        int         l_top_index = lua_gettop(_l_state);
    #endif

    l_status    = lua_load(_l_state, input_reader, in_data, input_source_name.c_str());
    l_response  = lua_tostring(_l_state, -1);

    if (l_status != 0) {
        if (l_status == LUA_ERRSYNTAX) {
            // check if command is incomplete
            if (std::string(l_response).find(_l_incomplete_input_msg) != std::string::npos) {
                lua_pop(_l_state, 1);
                return (SCRIPT_INCOMPLETE_INPUT);
            }
            // output error message from lua
            else {
                console.get() << l_response << std::endl;
                return (SCRIPT_SYNTAX_ERROR);
            }
        }
        else { // l_status == LUA_ERRMEM
            console.get() << "script_system_lua::process_script(): (error)"
                          << "memory error while loading script (lua_load() returned LUA_ERRMEM)" << std::endl;
            return (SCRIPT_MEMORY_ERROR);
        }
    }

    l_status    = lua_pcall(_l_state, 0, 0, 0);
    l_response  = lua_tostring(_l_state, -1);

    // output error message from lua
    if (l_status != 0) {
        console.get() << l_response << std::endl;
        lua_pop(_l_state, 1);
        return (SCRIPT_RUNTIME_ERROR);
    }

    #ifdef SCM_DEBUG
        assert(l_top_index == lua_gettop(_l_state));
    #endif

    // everything went fine
    return (SCRIPT_NO_ERROR);
}

#endif

#include <scm/core/utilities/luabind_warning_enable.h>
