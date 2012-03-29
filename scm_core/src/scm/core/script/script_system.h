
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCRIPT_SYSTEM_H_INCLUDED
#define SCRIPT_SYSTEM_H_INCLUDED

#if 0
#include <istream>
#include <ostream>
#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

#include <scm/core/sys_interfaces.h>

namespace scm {
namespace script {

typedef enum
{
    SCRIPT_NO_ERROR             = 0x00,
    SCRIPT_SYNTAX_ERROR,
    SCRIPT_RUNTIME_ERROR,
    SCRIPT_MEMORY_ERROR,
    SCRIPT_UNKNOWN_ERROR,
    SCRIPT_INCOMPLETE_INPUT
} script_result_t;

class __scm_export(core) script_system_interface : public scm::system_refreshable
{
public:
    typedef double      number_t;

public:
    script_system_interface();
    virtual ~script_system_interface();

    using scm::system_refreshable::initialize;
    using scm::system_refreshable::shutdown;
    using scm::system_refreshable::frame;       // should not be lua dependant

    script_result_t             do_script       (const std::string& script,
                                                 const std::string& input_source_name = std::string("unnamed string input source"));
    script_result_t             do_script       (std::istream& script,
                                                 const std::string& input_source_name = std::string("unnamed stream input source"));
    script_result_t             do_script_file  (const std::string& script_file);

    script_result_t             interpret_script(const std::string& script,
                                                 const std::string& input_source_name = std::string("unnamed string input source"));
    script_result_t             interpret_script(std::istream& script,
                                                 const std::string& input_source_name = std::string("unnamed stream input source"));

protected:
    virtual script_result_t     process_script  (std::istream& in_stream,
                                                 const std::string& input_source_name) = 0;
    virtual script_result_t     process_script  (const std::string& in_string,
                                                 const std::string& input_source_name) = 0;

private:
}; // class script_system_interface

} // namespace script
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // 0

#endif // SCRIPT_SYSTEM_H_INCLUDED
