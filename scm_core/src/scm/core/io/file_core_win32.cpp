
#include "file_core_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <algorithm>
#include <limits>
#include <map>
#include <queue>
#include <vector>
#include <cassert>

#include <scm/core/platform/windows.h>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/log.h>
#include <scm/core/math/math.h>
#include <scm/core/time/accumulate_timer.h>
#include <scm/core/time/high_res_timer.h>
#include <scm/core/utilities/foreach.h>

// TODO
// have one reader thread runnable/running which only sends out the read requests when one is put into the queue
// this way as soon as a request is free it can be resend if necessary, the calling thread only starts the read thread
// and waits for requests to be filled, after handling a filled request it is put back into the queue for open read
// requests and a condition variable for the read thread is triggered

namespace scm {
namespace io {

namespace detail {

struct overlapped_ext : public OVERLAPPED
{
    overlapped_ext(const file_core::size_type size);
    ~overlapped_ext(); // needs to be non-virtual

    void                                            position(const file_core::size_type pos);
    file_core::size_type                            position() const;

    void                                            bytes_to_process(const file_core::size_type size);
    file_core::size_type                            bytes_to_process() const;

    const scm::shared_ptr<file_core::char_type>&    buffer() const;

private:
    file_core::size_type                            _bytes_to_process;
    scm::shared_ptr<file_core::char_type>           _rw_buffer;
}; // struct overlapped_ext

typedef std::queue<request_ptr>                             request_ptr_queue;
typedef std::map<request_ptr::element_type*, request_ptr>   request_ptr_map;

overlapped_ext::overlapped_ext(const file_core::size_type size)
  : _bytes_to_process(0)
{
    hEvent = 0;
    _rw_buffer.reset(static_cast<file_core::char_type*>(VirtualAlloc(0, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)),
                                                        boost::bind<BOOL>(VirtualFree, _1, 0, MEM_RELEASE));
    assert(_rw_buffer);
}

overlapped_ext::~overlapped_ext()
{
    _rw_buffer.reset();
}

void
overlapped_ext::position(const file_core::size_type pos)
{
    LARGE_INTEGER   l;
    l.QuadPart = pos;

    this->Offset        = l.LowPart;
    this->OffsetHigh    = l.HighPart;
}

file_core::size_type
overlapped_ext::position() const
{
    LARGE_INTEGER   l;

    l.LowPart           = this->Offset;
    l.HighPart          = this->OffsetHigh;

    return (l.QuadPart);
}

void
overlapped_ext::bytes_to_process(const file_core::size_type size)
{
    _bytes_to_process = size;
}

file_core::size_type
overlapped_ext::bytes_to_process() const
{
    return (_bytes_to_process);
}

const scm::shared_ptr<file_core::char_type>&
overlapped_ext::buffer() const
{
    return (_rw_buffer);
}

struct io_result
{
    io_result() : _bytes_processed(0), _key(0), _ovl(0) {}

    DWORD           _bytes_processed;
    ULONG_PTR       _key;
    overlapped_ext* _ovl;
};

} // namespace detail

file_core_win32::file_core_win32()
{
}

file_core_win32::~file_core_win32()
{
}

bool
file_core_win32::open(const std::string&       file_path,
                      std::ios_base::openmode  open_mode,
                      bool                     disable_system_cache,
                      scm::uint32              read_write_buffer_size,
                      scm::uint32              read_write_asynchronous_requests)
{
    using namespace boost::filesystem;
#if 0
    path            input_file_path(file_path, native);
    path            complete_input_file_path(system_complete(input_file_path));

    bool            input_file_exists = exists(complete_input_file_path);
    std::string     input_root_path = complete_input_file_path.root_name();

    // translate open mode to access mode
    DWORD desired_access = 0;
    DWORD read_write_buffer_access = 0;

    if (open_mode & std::ios_base::in) {
        desired_access              |= GENERIC_READ;
        read_write_buffer_access    =  PAGE_READWRITE;
    }
    if (open_mode & std::ios_base::out) {
        desired_access              |= GENERIC_WRITE;
        read_write_buffer_access    =  PAGE_READWRITE;
    }

    // no async writes supported
    //if (   open_mode & std::ios_base::out
    //    && disable_system_cache)
    //{
    //    scm::err() << scm::log_level(scm::logging::ll_error)
    //               << "file_win::open(): "
    //               << "illegal open mode ("
    //               << std::hex << open_mode
    //               << ") write in compination with async read not supported"
    //               << "(on file '" << file_path << "')" << std::endl;

    //    return (false);
    //}

    // share mode
    DWORD share_mode = FILE_SHARE_READ;

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
            scm::err() << scm::log_level(scm::logging::ll_error)
                       << "file_win::open(): "
                       << "illegal open mode "
                       << std::hex << open_mode
                       << " on file '" << file_path << "'" << std::endl;

            return (false);
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
            scm::err() << scm::log_level(scm::logging::ll_error)
                       << "file_win::open(): "
                       << "illegal open mode "
                       << std::hex << open_mode
                       << " on file '" << complete_input_file_path.string() << "'" << std::endl;

            return (false);
        }
    }

    // file attributes
    DWORD flags_and_attributes = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN;

    if (disable_system_cache) {
        flags_and_attributes |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED;
    }

    // use shared pointer to manage the close in case we miss it somehow
    _file_handle.reset(CreateFile(complete_input_file_path.string().c_str(),
                                  desired_access,
                                  share_mode,
                                  0,
                                  creation_disposition,
                                  flags_and_attributes,
                                  0),
                       boost::bind<BOOL>(CloseHandle, _1));

    if (_file_handle.get() == INVALID_HANDLE_VALUE) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "file_win::open(): "
                   << "error creating/opening file:  "
                   << "'" << complete_input_file_path.string() << "'" << std::endl;

        return (false);
    }

    if (disable_system_cache) {
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

            scm::err() << scm::log_level(scm::logging::ll_error)
                       << "file_win::open(): "
                       << "error retrieving sector size information "
                       << "on device of file '" << complete_input_file_path.string() << "'" << std::endl;

            return (false);
        }

        _volume_sector_size = bytes_per_sector;

        assert(_volume_sector_size != 0);

        // create completion io port for asynchronous read/write operations
        _completion_port.reset(CreateIoCompletionPort(_file_handle.get(), NULL, 0 /*generate key*/, 0),
                               boost::bind<BOOL>(CloseHandle, _1));

        if (_completion_port.get() == NULL) {
            scm::err() << scm::log_level(scm::logging::ll_error)
                       << "file_win::open(): "
                       << "error creating completion io port:  "
                       << "'" << complete_input_file_path.string() << "'" << std::endl;

            return (false);
        }

        // calculate the correct read write buffer size (round up to full multiple of bytes per sector)
        _async_request_buffer_size  = static_cast<scm::int32>(vss_align_ceil(read_write_buffer_size));
        _async_requests             = read_write_asynchronous_requests;

        assert(_async_request_buffer_size % _volume_sector_size == 0);
        assert(read_write_buffer_access != 0);
    }

    if (   open_mode & std::ios_base::ate
        || open_mode & std::ios_base::app) {

        _position = actual_file_size();
    }

    _file_path  = complete_input_file_path.string();
    _open_mode  = open_mode;
    _file_size  = actual_file_size();
#endif
    return (true);
}

bool
file_core_win32::is_open() const
{
    if (_file_handle) {
        return ((_file_handle.get() == INVALID_HANDLE_VALUE) ? false : true);
    }
    else {
        return (false);
    }
}

void
file_core_win32::close()
{
}

file_core_win32::size_type
file_core_win32::read(char_type*const output_buffer,
                      size_type       num_bytes_to_read)
{
    return (0);
}

file_core_win32::size_type
file_core_win32::write(const char_type*const input_buffer,
                       size_type             num_bytes_to_write)
{
    return (0);
}

file_core_win32::size_type
file_core_win32::set_end_of_file()
{
    return (0);
}

} // namepspace io
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
