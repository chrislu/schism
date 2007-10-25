
#include <scm/core/math/math.h>

#include <iostream>
#include <string>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/data/volume/volume_data_loader_vgeo.h>
#include <scm/data/volume/volume_data_loader_raw.h>
#include <scm/data/volume/volume_data_loader_svol.h>

#include <scm/data/volume/scm_vol/scm_vol.h>

unsigned        max_dim_x;
unsigned        max_dim_y;
unsigned        max_dim_z;

std::string     input_file;
std::string     output_file;

int main(int argc, char **argv)
{
    try {
        boost::program_options::options_description  cmd_options("program options");

        cmd_options.add_options()
            ("help", "show this help message")
            ("mx", boost::program_options::value<unsigned>(&max_dim_x)->default_value(512), "maximal volume texture resolution x axis")
            ("my", boost::program_options::value<unsigned>(&max_dim_y)->default_value(512), "maximal volume texture resolution x axis")
            ("mz", boost::program_options::value<unsigned>(&max_dim_z)->default_value(512), "maximal volume texture resolution x axis")
            ("i", boost::program_options::value<std::string>(&input_file), "input volume file (.vol, .raw)")
            ("o", boost::program_options::value<std::string>(&output_file), "output volume file (.svol)");

        boost::program_options::variables_map           command_line;
        boost::program_options::parsed_options          parsed_cmd_line =  boost::program_options::parse_command_line(argc, argv, cmd_options);

        boost::program_options::store(parsed_cmd_line, command_line);
        boost::program_options::notify(command_line);

        if (command_line.count("help")) {
            std::cout << "usage: " << std::endl;
            std::cout << cmd_options;
            return (0);
        }
        if (   !command_line.count("i")
            || !command_line.count("o")) {
            std::cout << "<error>: no input or output file specified!" << std::endl;
            std::cout << "usage: " << std::endl;
            std::cout << cmd_options;
            return (0);
        }
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return (-1);
    }

    using namespace boost::filesystem;

    path                    ifile_path(input_file);
    std::string             ifile_name       = ifile_path.leaf();
    std::string             ifile_extension  = extension(ifile_path);

    path                    ofile_path(output_file);
    std::string             ofile_name       = ofile_path.leaf();
    std::string             ofile_extension  = extension(ofile_path);

    bool                    overwrite_ofile  = false;

    if (ofile_extension != ".svol") {
        ofile_path = ofile_path.branch_path() / (basename(ofile_path) + ".svol");
        ofile_extension  = extension(ofile_path);
    }

    std::cout << "input file:\t\t" << ifile_path.file_string() << std::endl
              << "output file:\t\t" << ofile_path.file_string() << std::endl
              << "volume dimensions:\t(" << max_dim_x << ", " << max_dim_y << ", " << max_dim_z << ")" << std::endl << std::endl;

    // check if input file exists and has valid extension
    if (!exists(ifile_path) || is_directory(ifile_path)) {
        std::cout << "<error> input file ('" << ifile_path.file_string() << "') "
                  << "does not exist or is a directory" << std::endl;
        return (-1);
    }

    if (   ifile_extension != ".vol"
        && ifile_extension != ".raw") {
        std::cout << "<error> input file ('" << ifile_path.file_string() << "') "
                  << "is not of a supported volume type (.raw, .vol)" << std::endl;
        return (-1);
    }

    // now check if output file exists, i yes ask i overwrite is ok
    if (is_directory(ofile_path)) {
        std::cout << "<error> output file ('" << ofile_path.file_string() << "') "
                  << "is a directory" << std::endl;
        return (-1);
    }
    if (exists(ofile_path)) {
        std::cout << "output file ('" << ofile_path.file_string() << "') "
                  << "allready exists" << std::endl;

        std::cout << "overwrite? (y/n): ";
        char a;
        std::cin  >> a;

        if (a == 'y' || a == 'Y') {
            overwrite_ofile = true;
        }
        else {
            std::cout << std::endl << "please choose another output file" << std::endl;
            return (-1);
        }
    }

    boost::scoped_ptr<scm::data::volume_data_loader> vol_loader;    
    if (ifile_extension == ".vol") {
        vol_loader.reset(new scm::data::volume_data_loader_vgeo());
    }
    else if (ifile_extension == ".raw") {
        vol_loader.reset(new scm::data::volume_data_loader_raw());
    }
    else if (ifile_extension == ".svol") {
        vol_loader.reset(new scm::data::volume_data_loader_svol());
    }

    if (!vol_loader->open_file(ifile_path.file_string())) {
        std::cout << "<error> during opening of file ('" << ifile_path.file_string() << "')"
                  << std::endl;
        return (-1);
    }

    const scm::data::volume_descriptor&  vol_desc = vol_loader->get_volume_descriptor();

    math::vec3ui_t data_dimensions = vol_desc._data_dimensions;

    std::cout << std::endl
              << "input volume data dimensions: "
              << "("  << data_dimensions.x
              << ", " << data_dimensions.y
              << ", " << data_dimensions.z << ")" << std::endl;

    unsigned int tiles_x, tiles_y, tiles_z;

    tiles_x = unsigned(math::ceil(float(data_dimensions.x) / float(max_dim_x)));
    tiles_y = unsigned(math::ceil(float(data_dimensions.y) / float(max_dim_y)));
    tiles_z = unsigned(math::ceil(float(data_dimensions.z) / float(max_dim_z)));

    std::cout << std::endl
              << "outputting volume tiles (num_x, num_y, num_z): "
              << "("  << tiles_x
              << ", " << tiles_y
              << ", " << tiles_z << ")" << std::endl;

    unsigned off_x, off_y, off_z;
    unsigned d_x, d_y, d_z;
    unsigned rem_x, rem_y, rem_z;

    rem_x = data_dimensions.x % max_dim_x;
    rem_y = data_dimensions.y % max_dim_y;
    rem_z = data_dimensions.z % max_dim_z;
    
    boost::scoped_array<unsigned char> buffer;

    std::string out_filename_svol;
    std::string out_filename_sraw;

    std::ofstream out_file_svol;
    std::ofstream out_file_sraw;

    scm::data::volume_descriptor out_vol_desc;

    out_vol_desc._data_num_channels     = vol_desc._data_num_channels;
    out_vol_desc._data_byte_per_channel = vol_desc._data_byte_per_channel;
    out_vol_desc._volume_aspect         = vol_desc._volume_aspect;
    out_vol_desc._name                  = basename(ifile_path);

    for (unsigned z = 0; z < tiles_z; z++) {
        for (unsigned y = 0; y < tiles_y; y++) {
            for (unsigned x = 0; x < tiles_x; x++) {
                
                off_x = x * max_dim_x;
                off_y = y * max_dim_y;
                off_z = z * max_dim_z;

                d_x = (off_x + max_dim_x) > data_dimensions.x ? rem_x : max_dim_x;
                d_y = (off_y + max_dim_y) > data_dimensions.y ? rem_y : max_dim_y;
                d_z = (off_z + max_dim_z) > data_dimensions.z ? rem_z : max_dim_z;

                buffer.reset(new unsigned char[d_x * d_y * d_z]);

                if (!vol_loader->read_sub_volume_data(math::vec<unsigned, 3>(off_x, off_y, off_z),
                                                      math::vec<unsigned, 3>(d_x, d_y, d_z), buffer.get())) {
                    std::cout << "error reading data" << std::endl;
                    vol_loader->close_file();
                    return (-1);
                }

                out_vol_desc._data_dimensions = math::vec3ui_t(d_x, d_y, d_z);
                out_vol_desc._volume_origin   = vol_desc._volume_origin + math::vec3f_t(float(off_x), float(off_y), float(off_z));
                out_vol_desc._brick_index     = math::vec3ui_t(x, y, z);

                // do we have to put out multiple bricks
                if (tiles_x != 1 || tiles_y != 1 || tiles_z != 1) {
                    out_filename_svol =   (basename(ofile_path)
                                         + "_" + boost::lexical_cast<std::string>(x)
                                         + "_" + boost::lexical_cast<std::string>(y)
                                         + "_" + boost::lexical_cast<std::string>(z));

                    out_filename_sraw = out_filename_svol + ".sraw";
                    out_filename_svol = (ofile_path.branch_path() / (out_filename_svol + ofile_extension)).file_string();
                }
                else {
                    out_filename_svol = (ofile_path.branch_path() / (basename(ofile_path) + ofile_extension)).file_string();
                    out_filename_sraw = (basename(ofile_path) + ".sraw");
                }

                out_vol_desc._sraw_file = out_filename_sraw;

                out_filename_sraw = (ofile_path.branch_path() / out_filename_sraw).file_string();


                out_file_sraw.open(out_filename_sraw.c_str(), std::ios::out | std::ios::binary);
                if (!out_file_sraw) {
                    std::cout << "error opening output file: " << out_filename_sraw << std::endl;
                    vol_loader->close_file();
                    return (-1);
                }

                out_file_sraw.write(reinterpret_cast<const char*>(buffer.get()), d_x * d_y * d_z);

                if (out_file_sraw.fail()) {
                    out_file_sraw.close();
                    std::cout << "error writing output file: " << out_filename_sraw << std::endl;
                    vol_loader->close_file();
                    return (-1);
                }

                out_file_sraw.close();

                out_file_svol.open(out_filename_svol.c_str(), std::ios::out);
                if (!out_file_svol) {
                    std::cout << "error opening output file: " << out_filename_svol << std::endl;
                    vol_loader->close_file();
                    return (-1);
                }

                out_file_svol << out_vol_desc;

                if (out_file_svol.fail()) {
                    out_file_svol.close();
                    std::cout << "error writing output file: " << out_filename_svol << std::endl;
                    vol_loader->close_file();
                    return (-1);
                }

                out_file_svol.close();
            }
        }
    }

    vol_loader->close_file();

    return (0);
}