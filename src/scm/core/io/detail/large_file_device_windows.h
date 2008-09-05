
#ifndef SCM_IO_DETAIL_LARGE_FILE_DEVICE_WINDOWS_H_INCLUDED
#define SCM_IO_DETAIL_LARGE_FILE_DEVICE_WINDOWS_H_INCLUDED

#ifdef _WIN32

#include <WinNT.h>

#include <scm/core/int_types.h>
#include <scm/core/io/detail/large_file_device_impl.h>

namespace scm {
namespace io {
namespace detail {

template <typename char_type>
class large_file_device_windows : public large_file_device_impl<char_type>
{
public:
    large_file_device_windows();
    large_file_device_windows(const large_file_device_windows& rhs);
    virtual ~large_file_device_windows();

    std::streamsize         read(char_type* s,          std::streamsize n);
    std::streamsize         write(const char_type* s,   std::streamsize n);

    std::streampos          seek(boost::iostreams::stream_offset    off,
                                 std::ios_base::seek_dir            way);

    void                    open(const std::string&         file_path,
                                 std::ios_base::openmode    open_mode,
                                 bool                       disable_system_cache,
                                 scm::uint32                read_write_buffer_size);

    bool                    is_open() const;
    void                    close();

private:
    HANDLE                  _file_handle;
    scm::uint64             _current_position;

}; // class large_file_device_windows


} // namespace detail
} // namespace io
} // namespace scm

#include "large_file_device_windows.inl"

#endif // _WIN32

#endif // SCM_IO_DETAIL_LARGE_FILE_DEVICE_WINDOWS_H_INCLUDED
