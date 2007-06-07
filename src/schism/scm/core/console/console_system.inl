
namespace scm {
namespace con {

// define pass through std::ostream operator << to _output_buffer
template<class T> console_system& operator << (console_system& con, const T& rhs) {
    con._out_stream << rhs;
    return (con);
}

inline console_system& operator << (console_system& con, std::ostream& (*_Pfn)(std::ostream&)) {
    con._out_stream << _Pfn;
    return (con);
}

inline console_system& operator << (console_system& con, const log_level& level) {
    con._out_stream << level;
    return (con);
}

} // namespace con
} // namespace scm
