
#ifndef SCM_IO_DETAIL_LARGE_FILE_DEVICE_IMPL_H_INCLUDED
#define SCM_IO_DETAIL_LARGE_FILE_DEVICE_IMPL_H_INCLUDED

#include <iosfwd>

#include <boost/iostreams/positioning.hpp>

#include <scm/core/int_types.h>

namespace scm {
namespace io {
namespace detail {

template <typename char_type>
class large_file_device_impl
{
public:
    large_file_device_impl() {};
    large_file_device_impl(const large_file_device_impl& rhs) {};
    virtual ~large_file_device_impl() {};

    virtual std::streamsize         read(char_type* s,          std::streamsize n) = 0;
    virtual std::streamsize         write(const char_type* s,   std::streamsize n) = 0;

    virtual std::streampos          seek(boost::iostreams::stream_offset    off,
                                         std::ios_base::seek_dir            way) = 0;

    virtual void                    open(const std::string&         file_path,
                                         std::ios_base::openmode    open_mode,
                                         bool                       disable_system_cache,
                                         scm::uint32                read_write_buffer_size) = 0;

    virtual bool                    is_open() const = 0;
    virtual void                    close() = 0;

}; // class large_file_device_impl

} // namespace detail
} // namespace io
} // namespace scm

#endif // SCM_IO_DETAIL_LARGE_FILE_DEVICE_IMPL_H_INCLUDED
