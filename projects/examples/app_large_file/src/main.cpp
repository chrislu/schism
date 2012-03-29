
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

#include <scm/core.h>
#include <scm/log.h>
#include <scm/core/pointer_types.h>
#include <scm/core/io/file.h>
#include <scm/core/time/accum_timer.h>
#include <scm/core/time/high_res_timer.h>


int main(int argc, char **argv)
{
    // the usual
    std::ios_base::sync_with_stdio(false);
    scm::shared_ptr<scm::core>      scm_core(new scm::core(argc, argv));

    typedef scm::time::accum_timer<scm::time::high_res_timer>  timer_type;

    // what we now try is to generate a very large file
    //  - check if it is written correctly by reading it back!
    //  - check if it memory runs full if we use the system cache
    //  - check what difference it makes not using the file cache
    //  - check if regular flush operations change the cache fillup

    using namespace scm;
    using namespace scm::io;

    std::string         out_file_name  = "e:/data/large_file_test_out_00.data";
    shared_ptr<file>    out_file       = make_shared<file>();

    std::cout << "opening output file: " << out_file_name << std::endl;
    if (!out_file->open(out_file_name,
                        std::ios_base::out | std::ios_base::trunc,
                        true /*disable system cache*/,
                        512 * 1024)) {
        std::cerr << "unable to open ouput file: " << out_file_name << std::endl;
        return -1;
    }

    // the important constants ////////////////////////////////////////////////////////////////////
    file::size_type     out_block_size   = out_file->vss_align_ceil(512ll * 1024);    // KiB
    file::size_type     out_file_size    = 40ll * 1024 * 1024 * 1024;                 // GiB
    bool                out_random_write = true;

    std::cout << "out_block_size:   " << out_block_size << std::endl
              << "out_file_size:    " << out_file_size  << std::endl
              << "out_random_write: " << (out_random_write ? "true" : "false") << std::endl;

    shared_array<char>  out_buffer(new char[out_block_size]);

    boost::mt19937       rand_gen;
    boost::uniform_int<> rand_dist(0, 255);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(rand_gen, rand_dist);

    std::cout << "generating output block of random data" << std::endl;
    for (file::size_type i = 0; i < out_block_size; ++i) {
        out_buffer[i] = die();
    }

    // writing file ///////////////////////////////////////////////////////////////////////////////
    file::size_type     out_block_count = out_file_size / out_block_size;
    std::vector<file::size_type> out_positions;
    out_positions.reserve(out_block_count);

    for (file::size_type out_pos = 0; out_pos < out_block_count; ++out_pos) {
        out_positions[out_pos] = out_pos * out_block_size;
    }

    if (out_random_write) {
        std::random_shuffle(out_positions.begin(), out_positions.end());
    }

    timer_type          write_timer;
    std::cout << "starting to write file..." << std::endl;
    write_timer.start();
    for (file::size_type out_pos = 0; out_pos < out_block_count; ++out_pos) {
        if (out_file->write(out_buffer.get(), out_positions[out_pos], out_block_size) != out_block_size) {
            std::cerr << "error writing to file at position: " << out_pos * out_block_size << std::endl;
        }
        if (out_pos % (out_block_count / 100) == 0) {
            //write_timer.stop();
            //write_timer.start();
            std::cout << "writing file: " << std::fixed << 100.0 * (static_cast<double>(out_pos) / out_block_count) << "%";
            std::cout << "\xd";
        }
    }
    write_timer.stop();
    std::cout << "end writing file." << std::endl;

    out_file->close();

    double  write_time = time::to_seconds(write_timer.accumulated_duration());

    std::cout << "write time: " << std::fixed << std::setprecision(3) << write_time << "s, "
              << "write speed: " << (static_cast<double>(out_file_size) / (1024 * 1024)) / write_time << "MiB/s" << std::endl;


    // read back file and compare blocks //////////////////////////////////////////////////////////
    shared_ptr<file>    in_file       = make_shared<file>();

    std::cout << "opening input file: " << out_file_name << std::endl;
    if (!in_file->open(out_file_name,
                       std::ios_base::in,
                       true /*disable system cache*/,
                       512 * 1024)) {
        std::cerr << "unable to open input file: " << out_file_name << std::endl;
        return -1;
    }

    shared_array<char>  in_buffer(new char[out_block_size]);

    std::cout << "starting to reading file..." << std::endl;
    for (file::size_type in_pos = 0; in_pos < out_block_count; ++in_pos) {
        if (in_file->read(in_buffer.get(), in_pos * out_block_size, out_block_size) != out_block_size) {
            std::cerr << "error reading to file at position: " << in_pos * out_block_size << std::endl;
        }
        if (in_pos % (out_block_count / 100) == 0) {
            //write_timer.stop();
            //write_timer.start();
            std::cout << "reading file: " << std::fixed << 100.0 * (static_cast<double>(in_pos) / out_block_count) << "%";
            std::cout << "\xd";
        }
        for (file::size_type i = 0; i < out_block_size; ++i) {
            if (in_buffer[i] != out_buffer[i]) {
                std::cerr << "found error at file position: " << in_pos * out_block_size << std::endl;
            }
        }
    }
    std::cout << "end reading file." << std::endl;

    out_file->close();
    std::cout << "sick, sad world..." << std::endl;

    return 0;
}
