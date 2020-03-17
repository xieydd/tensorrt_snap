/*
 * @Author: xieydd
 * @since: 2020-03-14 11:06:30
 * @lastTime: 2020-03-14 11:58:59
 * @LastAuthor: Do not edit
 * @message: 
 */
#ifndef ARGS_PARSER_H
#define ARGS_PARSER_H

#include <string>
#include <vector>
#include <getopt.h>
#include <iostream>

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./test [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "If no data directories are given, the default is to use "
                 "(./data)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--batch          Run with batch.\n";
}

//!
//! /brief Struct to maintain command-line arguments.
//!
struct Args
{
    bool help{false};
    int useDLACore{-1};
    int batch{1};
    std::string dataDir;
};

//!
//! \brief Populates the Args struct with the provided command-line parameters.
//!
//! \throw invalid_argument if any of the arguments are not valid
//!
//! \return boolean If return value is true, execution can continue, otherwise program should exit
//!
inline bool parseArgs(Args &args, int argc, char *argv[])
{
    while (1)
    {
        int arg;
        static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"datadir", required_argument, 0, 'd'}, {"useDLACore", required_argument, 0, 'u'}, {"batch", required_argument, 0, 'b'}, {nullptr, 0, nullptr, 0}};
        int option_index = 0;
        arg = getopt_long(argc, argv, "hd:iu", long_options, &option_index);
        if (arg == -1)
        {
            break;
        }

        switch (arg)
        {
        case 'h':
            args.help = true;
            return true;
        case 'd':
            if (optarg)
            {
                args.dataDir = optarg;
            }
            else
            {
                std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                return false;
            }
            break;
        case 'u':
            if (optarg)
            {
                args.useDLACore = std::stoi(optarg);
            }
            break;
        case 'b':
            if (optarg)
            {
                args.batch = std::stoi(optarg);
            }
            break;
        default:
            return false;
        }
    }
    return true;
}

#endif // ARGS_PARSER_H