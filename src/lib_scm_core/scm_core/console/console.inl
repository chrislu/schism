
namespace scm
{
    namespace core
    {
        // define pass through std::ostream operator << to _output_buffer
        template<class T> console_interface& operator << (console_interface& con, const T& rhs) {
            (con._output_buffer) << rhs;
            return (con);
        }
        inline console_interface& operator << (console_interface& con, std::ostream& (__cdecl *_Pfn)(std::ostream&)) {
            (con._output_buffer) << _Pfn;
            return (con);
        }

    } // namespace core

} // namespace scm

