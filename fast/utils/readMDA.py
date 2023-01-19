#!/usr/bin/env python
# ----------------------------------------------------------------------- #
# Copyright (c) 2023, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2021. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# ----------------------------------------------------------------------- #
# version 4 Tim Mooney 8/22/02
# adapted for python3 + remove tkinter support : TZ 03/13/19

import os
import string
import sys
from xdrlib import *


class scanDim:
    def __init__(self):
        self.rank = 0
        self.dim = 0
        self.npts = 0
        self.curr_pt = 0
        self.scan_name = ""
        self.time = ""
        self.np = 0
        self.p = []  # list of scanPositioner instances
        self.nd = 0
        self.d = []  # list of scanDetector instances
        self.nt = 0
        self.t = []  # list of scanTrigger instances

    def __str__(self):
        if self.scan_name != "":
            s = '%dD data from "%s": %d/%d pts; %d pos\'s, %d dets, %d trigs' % (
                self.dim,
                self.scan_name,
                self.curr_pt,
                self.npts,
                self.np,
                self.nd,
                self.nt,
            )
        else:
            s = "%dD data (not read in)" % (self.dim)

        return s


class scanClass:
    def __init__(self):
        self.rank = 0
        self.npts = 0
        self.curr_pt = 0
        self.plower_scans = 0
        self.name = ""
        self.time = ""
        self.np = 0
        self.nd = 0
        self.nt = 0
        self.p = []
        self.d = []
        self.t = []


class scanPositioner:
    def __init__(self):
        self.number = 0
        self.fieldName = ""
        self.name = ""
        self.desc = ""
        self.step_mode = ""
        self.unit = ""
        self.readback_name = ""
        self.readback_desc = ""
        self.readback_unit = ""
        self.data = []

    def __str__(self):
        s = "positioner %d (%s), desc:%s, unit:%s\n" % (
            self.number,
            self.name,
            self.desc,
            self.unit,
        )
        s = s + '   step mode: %s, readback:"%s"\n' % (
            self.step_mode,
            self.readback_name,
        )
        s = s + "data:%s" % (str(self.data))
        return s


class scanDetector:
    def __init__(self):
        self.number = 0
        self.fieldName = ""
        self.name = ""
        self.desc = ""
        self.unit = ""
        self.data = []

    def __str__(self):
        s = "detector %d (%s), desc:%s, unit:%s, data:%s\n" % (
            self.number,
            self.name,
            self.desc,
            self.unit,
            str(self.data),
        )
        return s


class scanTrigger:
    def __init__(self):
        self.number = 0
        self.name = ""
        self.command = 0.0

    def __str__(self):
        s = "trigger %d (%s), command=%f\n" % (self.number, self.name, self.command)
        return s


def detName(i, new=0):
    """
    detName(i,new=0) - this function returns the detector name Di used in sscan record
    where
      i - specify the zero based detector sequence number
      new - 1 specify the version 5 Di names desired, default 0
    """
    if new:
        return "D%02d" % (i + 1)
    if i < 15:
        return "D%s" % (hex(i + 1)[2]).upper()
    elif i < 85:
        return "D%02d" % (i - 14)
    else:
        return "?"


def posName(i):
    """
    posName(i) - this function returns the positioner name Pi used in sscan record
    where
      i - specify the zero based positioner sequence number
    """
    if i < 4:
        return "P%d" % (i + 1)
    else:
        return "?"


def readScan(file, v, new=0):
    """
    readScan(file, v, new=0) - internal scan read routine, it unpack a subset of scan data from
    the current position of the file pointer
    it returns the scan data set extracted
    where
      file - file pointer of an opened MDA file
      v - input verbose specified
      new  - default 0, if 1 specified then version 5 Di name used
    """
    scan = scanClass()
    buf = file.read(10000)  # enough to read scan header
    u = Unpacker(buf)
    scan.rank = u.unpack_int()
    if v:
        print("scan.rank = ", repr(scan.rank))
    scan.npts = u.unpack_int()
    if v:
        print("scan.npts = ", repr(scan.npts))
    scan.curr_pt = u.unpack_int()
    if v:
        print("scan.curr_pt = ", repr(scan.curr_pt))
    if scan.rank > 1:
        # if curr_pt < npts, plower_scans will have garbage for pointers to
        # scans that were planned for but not written
        scan.plower_scans = u.unpack_farray(scan.npts, u.unpack_int)
        if v:
            print("scan.plower_scans = ", repr(scan.plower_scans))
    namelength = u.unpack_int()
    scan.name = u.unpack_string().decode()
    if v:
        print("scan.name = ", repr(scan.name))
    timelength = u.unpack_int()
    scan.time = u.unpack_string().decode()
    if v:
        print("scan.time = ", repr(scan.time))
    scan.np = u.unpack_int()
    if v:
        print("scan.np = ", repr(scan.np))
    scan.nd = u.unpack_int()
    if v:
        print("scan.nd = ", repr(scan.nd))
    scan.nt = u.unpack_int()
    if v:
        print("scan.nt = ", repr(scan.nt))
    for j in range(scan.np):
        scan.p.append(scanPositioner())
        scan.p[j].number = u.unpack_int()
        scan.p[j].fieldName = posName(scan.p[j].number)
        if v:
            print("positioner ", j)
        length = u.unpack_int()  # length of name string
        if length:
            scan.p[j].name = u.unpack_string().decode()
        if v:
            print("scan.p[%d].name = %s" % (j, repr(scan.p[j].name)))
        length = u.unpack_int()  # length of desc string
        if length:
            scan.p[j].desc = u.unpack_string().decode()
        if v:
            print("scan.p[%d].desc = %s" % (j, repr(scan.p[j].desc)))
        length = u.unpack_int()  # length of step_mode string
        if length:
            scan.p[j].step_mode = u.unpack_string().decode()
        if v:
            print("scan.p[%d].step_mode = %s" % (j, repr(scan.p[j].step_mode)))
        length = u.unpack_int()  # length of unit string
        try:
            if length:
                scan.p[j].unit = u.unpack_string().decode()
        except Exception as e:
            pass  # print(e)
        if v:
            print("scan.p[%d].unit = %s" % (j, repr(scan.p[j].unit)))
        length = u.unpack_int()  # length of readback_name string
        if length:
            scan.p[j].readback_name = u.unpack_string().decode()
        if v:
            print("scan.p[%d].readback_name = %s" % (j, repr(scan.p[j].readback_name)))
        length = u.unpack_int()  # length of readback_desc string
        if length:
            scan.p[j].readback_desc = u.unpack_string().decode()
        if v:
            print("scan.p[%d].readback_desc = %s" % (j, repr(scan.p[j].readback_desc)))
        length = u.unpack_int()  # length of readback_unit string
        if length:
            scan.p[j].readback_unit = u.unpack_string().decode()
        if v:
            print("scan.p[%d].readback_unit = %s" % (j, repr(scan.p[j].readback_unit)))

    for j in range(scan.nd):
        scan.d.append(scanDetector())
        scan.d[j].number = u.unpack_int()
        scan.d[j].fieldName = detName(scan.d[j].number, new=new)
        if v:
            print("detector ", j)
        length = u.unpack_int()  # length of name string
        if length:
            scan.d[j].name = u.unpack_string().decode()
        if v:
            print("scan.d[%d].name = %s" % (j, repr(scan.d[j].name)))
        length = u.unpack_int()  # length of desc string
        if length:
            scan.d[j].desc = u.unpack_string().decode()
        if v:
            print("scan.d[%d].desc = %s" % (j, repr(scan.d[j].desc)))
        length = u.unpack_int()  # length of unit string
        if length:
            scan.d[j].unit = u.unpack_string().decode()
        if v:
            print("scan.d[%d].unit = %s" % (j, repr(scan.d[j].unit)))

    for j in range(scan.nt):
        scan.t.append(scanTrigger())
        scan.t[j].number = u.unpack_int()
        if v:
            print("trigger ", j)
        length = u.unpack_int()  # length of name string
        if length:
            scan.t[j].name = u.unpack_string().decode()
        if v:
            print("scan.t[%d].name = %s" % (j, repr(scan.t[j].name)))
        scan.t[j].command = u.unpack_float()
        if v:
            print("scan.t[%d].command = %s" % (j, repr(scan.t[j].command)))

    ### read data

    # positioners
    file.seek(file.tell() - (len(buf) - u.get_position()))
    buf = file.read(scan.np * scan.npts * 8)
    u = Unpacker(buf)
    for j in range(scan.np):
        if v:
            print(
                "read %d pts for pos. %d at buf loc %x"
                % (scan.npts, j, u.get_position())
            )
        scan.p[j].data = u.unpack_farray(scan.npts, u.unpack_double)
        if v:
            print("scan.p[%d].data = %s" % (j, repr(scan.p[j].data)))

    # detectors
    file.seek(file.tell() - (len(buf) - u.get_position()))
    buf = file.read(scan.nd * scan.npts * 4)
    u = Unpacker(buf)
    for j in range(scan.nd):
        scan.d[j].data = u.unpack_farray(scan.npts, u.unpack_float)
        if v:
            print("scan.d[%d].data = %s" % (j, repr(scan.d[j].data)))

    return scan


def readMDA(fname=None, maxdim=2, verbose=1, help=0, new=0):
    """
    readMDA(fname=None, maxdim=2, verbose=1, help=0) - This fuction read an MDA file and
    construct the MDA data structure accordingly
    it returns the MDA data sturcture constructed
    where
      fname - specifies the input mda file name
      maxdim - specifies the max dimension extract, default 2
      verbose - reading info on or off, default 1
      help - echo help information on or off, default 0
      new - 1 specify the version 5 Di names desired, default 0

    e.g.

    from readMDA import *
    d = readMDA('/home/beams/CHA/data/xxx/cha_0001.mda')

    """
    dim = []

    if fname == None:
        pass
    if not os.path.isfile(fname):
        fname = fname + ".mda"
    if not os.path.isfile(fname):
        print(fname, " is not a file")
        return dim

    file = open(fname, "rb")
    # print "file = ", str(file)
    # file.seek(0,2)
    # filesize = file.tell()
    # file.seek(0)
    buf = file.read(100)  # to read header for scan of up to 5 dimensions
    u = Unpacker(buf)

    # read file header
    version = u.unpack_float()
    scan_number = u.unpack_int()
    rank = u.unpack_int()
    dimensions = u.unpack_farray(rank, u.unpack_int)
    isRegular = u.unpack_int()
    pExtra = u.unpack_int()
    pmain_scan = file.tell() - (len(buf) - u.get_position())

    for i in range(rank):
        dim.append(scanDim())
        dim[i].dim = i + 1
        dim[i].rank = rank - i

    file.seek(pmain_scan)
    s0 = readScan(file, max(0, verbose - 1), new=new)
    dim[0].npts = s0.npts
    dim[0].curr_pt = s0.curr_pt
    dim[0].scan_name = s0.name
    dim[0].time = s0.time
    dim[0].np = s0.np
    for i in range(s0.np):
        dim[0].p.append(s0.p[i])
    dim[0].nt = s0.nt
    for j in range(s0.nt):
        dim[0].t.append(s0.t[j])
    dim[0].nd = s0.nd
    for i in range(s0.nd):
        dim[0].d.append(s0.d[i])

    if (rank > 1) and (maxdim > 1):
        # collect 2D data
        for i in range(s0.curr_pt):
            file.seek(s0.plower_scans[i])
            s = readScan(file, max(0, verbose - 1), new=new)
            if i == 0:
                dim[1].npts = s.npts
                dim[1].curr_pt = s.curr_pt
                dim[1].scan_name = s.name
                dim[1].time = s.time
                # copy positioner, trigger, detector instances
                dim[1].np = s.np
                for j in range(s.np):
                    dim[1].p.append(s.p[j])
                    tmp = s.p[j].data[:]
                    dim[1].p[j].data = []
                    dim[1].p[j].data.append(tmp)
                dim[1].nt = s.nt
                for j in range(s.nt):
                    dim[1].t.append(s.t[j])
                dim[1].nd = s.nd
                for j in range(s.nd):
                    dim[1].d.append(s.d[j])
                    tmp = s.d[j].data[:]
                    dim[1].d[j].data = []
                    dim[1].d[j].data.append(tmp)
            else:
                # append data arrays
                for j in range(s.np):
                    dim[1].p[j].data.append(s.p[j].data)
                for j in range(s.nd):
                    dim[1].d[j].data.append(s.d[j].data)

    if (rank > 2) and (maxdim > 2):
        # collect 3D data
        for i in range(s0.curr_pt):
            file.seek(s0.plower_scans[i])
            s1 = readScan(file, max(0, verbose - 1), new=new)
            for j in range(s1.curr_pt):
                file.seek(s1.plower_scans[j])
                s = readScan(file, max(0, verbose - 1), new=new)
                if (i == 0) and (j == 0):
                    dim[2].npts = s.npts
                    dim[2].curr_pt = s.curr_pt
                    dim[2].scan_name = s.name
                    dim[2].time = s.time
                    # copy positioner, trigger, detector instances
                    dim[2].np = s.np
                    for k in range(s.np):
                        dim[2].p.append(s.p[k])
                        tmp = s.p[k].data[:]
                        dim[2].p[k].data = [[]]
                        dim[2].p[k].data[i].append(tmp)
                    dim[2].nt = s.nt
                    for k in range(s.nt):
                        dim[2].t.append(s.t[k])
                    dim[2].nd = s.nd
                    for k in range(s.nd):
                        dim[2].d.append(s.d[k])
                        tmp = s.d[k].data[:]
                        dim[2].d[k].data = [[]]
                        dim[2].d[k].data[i].append(tmp)
                elif j == 0:
                    for k in range(s.np):
                        dim[2].p[k].data.append([])
                        dim[2].p[k].data[i].append(s.p[k].data)
                    for k in range(s.nd):
                        dim[2].d[k].data.append([])
                        dim[2].d[k].data[i].append(s.d[k].data)
                else:
                    # append data arrays
                    for k in range(s.np):
                        dim[2].p[k].data[i].append(s.p[k].data)
                    for k in range(s.nd):
                        dim[2].d[k].data[i].append(s.d[k].data)

    # Collect scan-environment variables into a dictionary
    dict = {}
    dict["sampleEntry"] = ("description", "unit string", "value")
    dict["filename"] = fname
    dict["rank"] = rank
    dict["dimensions"] = dimensions
    if pExtra:
        file.seek(pExtra)
        buf = file.read()  # Read all scan-environment data
        u = Unpacker(buf)
        numExtra = u.unpack_int()
        for i in range(numExtra):
            name = ""
            n = u.unpack_int()  # length of name string
            if n:
                name = u.unpack_string().decode()
            desc = ""
            n = u.unpack_int()  # length of desc string
            if n:
                desc = u.unpack_string().decode()
            type = u.unpack_int()

            unit = ""
            value = ""
            count = 0
            if type != 0:  # not DBR_STRING
                count = u.unpack_int()  #
                n = u.unpack_int()  # length of unit string
                if n:
                    unit = u.unpack_string().decode()

            if type == 0:  # DBR_STRING
                n = u.unpack_int()  # length of value string
                if n:
                    value = u.unpack_string().decode()
            elif type == 32:  # DBR_CTRL_CHAR
                # value = u.unpack_fstring(count)
                v = u.unpack_farray(count, u.unpack_int)
                value = ""
                for i in range(len(v)):
                    # treat the byte array as a null-terminated string
                    if v[i] == 0:
                        break
                    value = value + chr(v[i])

            elif type == 29:  # DBR_CTRL_SHORT
                value = u.unpack_farray(count, u.unpack_int)
            elif type == 33:  # DBR_CTRL_LONG
                value = u.unpack_farray(count, u.unpack_int)
            elif type == 30:  # DBR_CTRL_FLOAT
                value = u.unpack_farray(count, u.unpack_float)
            elif type == 34:  # DBR_CTRL_DOUBLE
                value = u.unpack_farray(count, u.unpack_double)

            dict[name] = (desc, unit, value)

    dim.reverse()
    dim.append(dict)
    dim.reverse()
    if verbose:
        print(
            "%s is a %d-D file; %d dimensions read in."
            % (fname, dim[0]["rank"], len(dim) - 1)
        )
        print("dim[0] = dictionary of %d scan-environment PVs" % (len(dim[0])))
        print("   usage: dim[0]['sampleEntry'] ->", dim[0]["sampleEntry"])
        for i in range(1, len(dim)):
            print("dim[%d] = %s" % (i, str(dim[i])))
        print("   usage: dim[1].p[2].data -> 1D array of positioner 2 data")
        print("   usage: dim[2].d[7].data -> 2D array of detector 7 data")

    if help:
        print(" ")
        print("   each dimension (e.g., dim[1]) has the following fields: ")
        print(
            "      time      - date & time at which scan was started: %s"
            % (dim[1].time)
        )
        print(
            "      scan_name - name of scan record that acquired this dimension: '%s'"
            % (dim[1].scan_name)
        )
        print(
            "      curr_pt   - number of data points actually acquired: %d"
            % (dim[1].curr_pt)
        )
        print("      npts      - number of data points requested: %d" % (dim[1].npts))
        print(
            "      nd        - number of detectors for this scan dimension: %d"
            % (dim[1].nd)
        )
        print("      d[]       - list of detector-data structures")
        print(
            "      np        - number of positioners for this scan dimension: %d"
            % (dim[1].np)
        )
        print("      p[]       - list of positioner-data structures")
        print(
            "      nt        - number of detector triggers for this scan dimension: %d"
            % (dim[1].nt)
        )
        print("      t[]       - list of trigger-info structures")

    if help:
        print(" ")
        print(
            "   each detector-data structure (e.g., dim[1].d[0]) has the following fields: "
        )
        print("      desc      - description of this detector")
        print("      data      - data list")
        print("      unit      - engineering units associated with this detector")
        print("      fieldName - scan-record field (e.g., 'D01')")

    if help:
        print(" ")
        print(
            "   each positioner-data structure (e.g., dim[1].p[0]) has the following fields: "
        )
        print("      desc          - description of this positioner")
        print("      data          - data list")
        print("      step_mode     - scan mode (e.g., Linear, Table, On-The-Fly)")
        print("      unit          - engineering units associated with this positioner")
        print("      fieldName     - scan-record field (e.g., 'P1')")
        print("      name          - name of EPICS PV (e.g., 'xxx:m1.VAL')")
        print("      readback_desc - description of this positioner")
        print("      readback_unit - engineering units associated with this positioner")
        print("      readback_name - name of EPICS PV (e.g., 'xxx:m1.VAL')")

    return dim


def writeMDA(dim, fname=None):
    raise NotImplementedError
    m = mdaBuf()
    p = xdr.Packer()

    p.reset()
    if type(dim) != type([]):
        print("writeMDA: first arg must be a scan")
    if (fname != None) and (type(fname) != type("")):
        print("writeMDA: second arg must be a filename or None")
    rank = dim[0]["rank"]  # rank of scan as a whole
    # write file header
    p.pack_float(dim[0]["version"])
    p.pack_int(dim[0]["scan_number"])
    p.pack_int(dim[0]["rank"])
    p.pack_farray(rank, dim[0]["dimensions"], p.pack_int)
    p.pack_int(dim[0]["isRegular"])
    m.header = p.get_buffer()

    p.reset()
    p.pack_int(0)  # pExtra
    m.pExtra = p.get_buffer()

    m.scan = packScanHead(dim[1])
    m.scan.offset = len(m.header) + len(m.pExtra)
    m.scan.data = packScanData(dim[1], [])
    m.scan.bufLen = m.scan.bufLen + len(m.scan.data)
    prevScan = m.scan
    # print "\n m.scan=", m.scan
    # print "\n type(m.scan.pLowerScans)=", type(m.scan.pLowerScans)

    if rank > 1:
        for i in range(m.scan.npts):
            m.scan.inner.append(packScanHead(dim[2]))
            thisScan = m.scan.inner[i]
            thisScan.offset = prevScan.offset + prevScan.bufLen
            m.scan.pLowerScans.append(thisScan.offset)
            thisScan.data = packScanData(dim[2], [i])
            thisScan.bufLen = thisScan.bufLen + len(thisScan.data)
            prevScan = thisScan

            if rank > 2:
                for j in range(m.scan.inner[i].npts):
                    m.scan.inner[i].inner.append(packScanHead(dim[3]))
                    thisScan = m.scan.inner[i].inner[j]
                    thisScan.offset = prevScan.offset + prevScan.bufLen
                    m.scan.inner[i].pLowerScans.append(thisScan.offset)
                    thisScan.data = packScanData(dim[3], [i, j])
                    thisScan.bufLen = thisScan.bufLen + len(thisScan.data)
                    prevScan = thisScan

                if rank > 3:
                    for k in range(m.scan.inner[i].inner[j].npts):
                        m.scan.inner[i].inner[j].append(packScanHead(dim[4]))
                        thisScan = m.scan.inner[i].inner[j].inner[k]
                        thisScan.offset = prevScan.offset + prevScan.bufLen
                        m.scan.inner[i].inner[j].pLowerScans.append(thisScan.offset)
                        thisScan.data = packScanData(dim[4], [i, j, k])
                        thisScan.bufLen = thisScan.bufLen + len(thisScan.data)
                        prevScan = thisScan

    # Now we know where the extraPV section must go.
    p.reset()
    p.pack_int(prevScan.offset + prevScan.bufLen)  # pExtra
    m.pExtra = p.get_buffer()

    # pack scan-environment variables from dictionary
    p.reset()

    numKeys = 0
    for name in dim[0].keys():
        if not (name in dim[0]["ourKeys"]):
            numKeys = numKeys + 1
    p.pack_int(numKeys)

    for name in dim[0].keys():
        # Note we don't want to write the dict entries we made for our own
        # use in the scanDim object.
        if not (name in dim[0]["ourKeys"]):
            desc = dim[0][name][0]
            unit = dim[0][name][1]
            value = dim[0][name][2]
            EPICS_type = dim[0][name][3]
            count = dim[0][name][4]
            n = len(name)
            p.pack_int(n)
            if n:
                p.pack_string(name)
            n = len(desc)
            p.pack_int(n)
            if n:
                p.pack_string(desc)
            p.pack_int(EPICS_type)
            if EPICS_type != 0:  # not DBR_STRING, so pack count and units
                p.pack_int(count)
                n = len(unit)
                p.pack_int(n)
                if n:
                    p.pack_string(unit)
            if EPICS_type == 0:  # DBR_STRING
                n = len(value)
                p.pack_int(n)
                if n:
                    p.pack_string(value)
            elif EPICS_type == 32:  # DBR_CTRL_CHAR
                # write null-terminated string
                v = []
                for i in range(len(value)):
                    v.append(ord(value[i : i + 1]))
                v.append(0)
                p.pack_farray(count, v, p.pack_int)
            elif EPICS_type == 29:  # DBR_CTRL_SHORT
                p.pack_farray(count, value, p.pack_int)
            elif EPICS_type == 33:  # DBR_CTRL_LONG
                p.pack_farray(count, value, p.pack_int)
            elif EPICS_type == 30:  # DBR_CTRL_FLOAT
                p.pack_farray(count, value, p.pack_float)
            elif EPICS_type == 34:  # DBR_CTRL_DOUBLE
                p.pack_farray(count, value, p.pack_double)

    m.extraPV = p.get_buffer()

    # Now we have to repack all the scan offsets
    if rank > 1:  # 2D scan
        # print "m.scan.pLowerScans", m.scan.pLowerScans
        p.reset()
        p.pack_farray(m.scan.npts, m.scan.pLowerScans, p.pack_int)
        m.scan.pLowerScansBuf = p.get_buffer()
        if rank > 2:  # 3D scan
            for i in range(m.scan.npts):
                p.reset()
                p.pack_farray(
                    m.scan.inner[i].npts, m.scan.inner[i].pLowerScans, p.pack_int
                )
                m.scan.inner[i].pLowerScansBuf = p.get_buffer()
                if rank > 3:  # 4D scan
                    for j in range(m.scan.inner[i].npts):
                        p.reset()
                        p.pack_farray(
                            m.scan.inner[i].inner[j].npts,
                            m.scan.inner[i].inner[j].pLowerScans,
                            p.pack_int,
                        )
                        m.scan.inner[i].inner[j].pLowerScansBuf = p.get_buffer()

    # Write
    if fname == None:
        fname = tkFileDialog.SaveAs().show()
    f = open(fname, "wb")

    f.write(m.header)
    f.write(m.pExtra)
    s0 = m.scan
    f.write(s0.preamble)
    if len(s0.pLowerScansBuf):
        f.write(s0.pLowerScansBuf)
    f.write(s0.postamble)
    f.write(s0.data)
    for s1 in s0.inner:
        f.write(s1.preamble)
        if len(s1.pLowerScansBuf):
            f.write(s1.pLowerScansBuf)
        f.write(s1.postamble)
        f.write(s1.data)
        for s2 in s1.inner:
            f.write(s2.preamble)
            if len(s2.pLowerScansBuf):
                f.write(s2.pLowerScansBuf)
            f.write(s2.postamble)
            f.write(s2.data)
            for s3 in s2.inner:
                f.write(s3.preamble)
                if len(s3.pLowerScansBuf):
                    f.write(s3.pLowerScansBuf)
                f.write(s3.postamble)
                f.write(s3.data)
    f.write(m.extraPV)
    f.close()
    return
