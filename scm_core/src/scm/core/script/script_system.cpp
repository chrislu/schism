
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "script_system.h"

//#include <scm/core/core.h>

#if 0
#include <scm/core/console/console.h>

#include <fstream>

using namespace scm::core;

script_system_interface::script_system_interface()
{
}

script_system_interface::~script_system_interface()
{
}

script_result_t script_system_interface::do_script(const std::string& script,
                                                   const std::string& input_source_name)
{
    return (process_script(script, input_source_name));
}

script_result_t script_system_interface::do_script(std::istream& script,
                                                   const std::string& input_source_name)
{
    return (process_script(script, input_source_name));
}

script_result_t script_system_interface::do_script_file(const std::string& script_file)
{
    std::ifstream   file;

    file.open(script_file.c_str(), std::ios::binary);

    if (!file.is_open()) {
        console.get() << "script_system::do_script_file(): (error) unable to open file: ";
        console.get() << script_file << std::endl;

        return (SCRIPT_UNKNOWN_ERROR);
    }
    
    script_result_t ret = process_script(file, std::string("file: ") + script_file);

    file.close();

    return (ret);
}

script_result_t script_system_interface::interpret_script(const std::string& script,
                                                          const std::string& input_source_name)
{
    static std::string          complete_input;
    script_result_t             run_result;

    complete_input += script;
    run_result      = do_script(complete_input, input_source_name);

    if (run_result != SCRIPT_INCOMPLETE_INPUT){
        complete_input.clear();
    }

    return (run_result);
}

script_result_t script_system_interface::interpret_script(std::istream& script,
                                                          const std::string& input_source_name)
{
    return (SCRIPT_SYNTAX_ERROR);
}

#endif
