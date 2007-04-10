
#ifndef CONSOLE_H_INCLUDED
#define CONSOLE_H_INCLUDED

#include <sstream>
#include <string>
#include <deque>

#include <boost/noncopyable.hpp>

#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

//#if SCM_CORE_BUILD
//template class __scm_export std::basic_stringstream<char, std::char_traits<char>, std::allocator<char> >;
//#else
//template class __scm_export std::basic_stringstream<char, std::char_traits<char>, std::allocator<char> >;
//#endif

namespace scm
{
    namespace core
    {
        //class c_var;

        class __scm_export console_interface : boost::noncopyable
        {
        public:
            typedef std::deque<std::string>     input_history_container;

        public:
            console_interface(input_history_container::size_type hs = 50);
            virtual ~console_interface();

            void                                add_input(const std::string&);
            bool                                process_input();

        protected:
            std::stringstream                   _output_buffer;
            std::stringstream                   _input_buffer;

            input_history_container             _input_history;
            input_history_container::size_type  _input_history_max_length;

            void                                add_input_history(const std::string&);

     private:
            // pass through std::ostream operator << to _output_buffer
            template <class T>
            friend console_interface& operator << (console_interface& con, const T& rhs);
            friend console_interface& operator << (console_interface& con, std::ostream& (__cdecl *_Pfn)(std::ostream&));

        }; // class console_interface

    } // namespace core

} // namespace scm

#include "console.inl"

#include <scm_core/utilities/platform_warning_enable.h>

#endif // CONSOLE_H_INCLUDED
