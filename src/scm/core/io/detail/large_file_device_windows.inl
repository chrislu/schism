
#include <windows.h>

#include <boost/filesystem.hpp>

namespace scm {
namespace io {
namespace detail {

template <typename char_type>
large_file_device_windows<char_type>::large_file_device_windows()
  : _file_handle(INVALID_HANDLE_VALUE),
    _current_position(0)
{
}

template <typename char_type>
large_file_device_windows<char_type>::large_file_device_windows(const large_file_device_windows<char_type>& rhs)
  : _file_handle(rhs._file_handle),
    _current_position(rhs._current_position)
{
}

template <typename char_type>
large_file_device_windows<char_type>::~large_file_device_windows()
{
}

template <typename char_type>
std::streamsize
large_file_device_windows<char_type>::read(char_type* s, std::streamsize n)
{
    // TODO
    return (0);
}

template <typename char_type>
std::streamsize
large_file_device_windows<char_type>::write(const char_type* s, std::streamsize n)
{
    // TODO
    return (0);
}

template <typename char_type>
std::streampos
large_file_device_windows<char_type>::seek(boost::iostreams::stream_offset    off,
                                           std::ios_base::seek_dir            way)
{
    // TODO
    return (0);
}

template <typename char_type>
void
large_file_device_windows<char_type>::open(const std::string&         file_path,
                                           std::ios_base::openmode    open_mode,
                                           bool                       disable_system_cache,
                                           scm::uint32                read_write_buffer_size)
{
    using namespace boost::filesystem;

    path            input_file_path(file_path, native);
    path            complete_input_file_path(system_complete(input_file_path));

    bool            input_file_exists = exists(complete_input_file_path);
    std::string     input_root_path = complete_input_file_path.root_name();

    // translate open mode to access mode
    DWORD desired_access = 0;

    if (open_mode & std::ios_base::in) {
        desired_access |= GENERIC_READ;
    }
    if (open_mode & std::ios_base::out) {
        desired_access |= GENERIC_WRITE;
    }

    // share mode
    DWORD share_mode = 0;

    // translate open mode to creation modes
    DWORD creation_disposition = 0;

    if (input_file_exists) {
        if (    (open_mode & std::ios_base::out
              || open_mode & std::ios_base::in)
            && !(open_mode & std::ios_base::trunc)) {
            creation_disposition = OPEN_ALWAYS;
        }
        else if (   open_mode & std::ios_base::out
                 && open_mode & std::ios_base::trunc) {
            creation_disposition = TRUNCATE_EXISTING;
        }
        else {
            throw std::ios_base::failure("large_file_device_windows<char_type>::open(): illegal open mode");
        }
    }
    else {
        if (    (open_mode & std::ios_base::out
              || open_mode & std::ios_base::in)
            && !(open_mode & std::ios_base::trunc)) {
            creation_disposition = CREATE_NEW;
        }
        else if (   open_mode & std::ios_base::out
                 && open_mode & std::ios_base::trunc) {
            creation_disposition = CREATE_NEW;
        }
        else {
            throw std::ios_base::failure("large_file_device_windows<char_type>::open(): illegal open mode");
        }
    }

    // file attributes
    DWORD flags_and_attributes = FILE_ATTRIBUTE_NORMAL;

    if (disable_system_cache) {
        flags_and_attributes |= FILE_FLAG_NO_BUFFERING;

        // retrieve the sector size information
        DWORD   sectors_per_cluster;
        DWORD   bytes_per_sector;
        DWORD   free_clusters;
        DWORD   total_clusters;

        if (GetDiskFreeSpace(input_root_path.c_str(),
                             &sectors_per_cluster,
                             &bytes_per_sector,
                             &free_clusters,
                             &total_clusters) == FALSE) {
            throw std::ios_base::failure("large_file_device_windows<char_type>::open(): error retrieving sector size information");
        }

        // calculate the correct read write buffer size (round up to full multiple of bytes per sector)
        scm::uint32         rw_buf_size = ((read_write_buffer_size / bytes_per_sector) + 1) * bytes_per_sector;

        // TODO initialize read_write_buffer;
    }

    _file_handle = CreateFile(complete_input_file_path.string().c_str(),
                              desired_access,
                              share_mode,
                              0,
                              creation_disposition,
                              flags_and_attributes,
                              0);

    if (_file_handle == INVALID_HANDLE_VALUE) {
        throw std::ios_base::failure(  std::string("large_file_device_windows<char_type>::open(): error creating/opening file ")
                                     + complete_input_file_path.string());
    }
}

template <typename char_type>
bool
large_file_device_windows<char_type>::is_open() const
{
    return (_file_handle != INVALID_HANDLE_VALUE);
}

template <typename char_type>
void
large_file_device_windows<char_type>::close()
{
    if (is_open()) {
        CloseHandle(_file_handle);
        _file_handle = INVALID_HANDLE_VALUE;
    }
}

} // namespace detail
} // namespace io
} // namespace scm
