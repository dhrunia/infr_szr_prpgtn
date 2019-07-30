#!/usr/bin/env python


# This is licensed under a non-disclosure agreement to clinical use
# at the INS, MEG, APHM etc. Not to be distributed under any conditions!

from __future__ import division
import os
import sys
import numpy as np
import scipy.io


MAXELEC = 1024
OLDMAXELEC = 128
MAXGRP = 5
MAXSPC = 16
MAXVOIESMTG = 512
OLDMAXVOIESMTG = 64
UNCRYPT = 0
CRYPT = 1

infoMontage = np.dtype([
    ('originalMontageFileName', 'S20'),
    ('originalElectrodsFileName', 'S20'),
    ('nbMontageUsed', 'H'),
])

header = np.dtype([
    ('codeBlock', 'I'),
    ('blockLen', 'I'),
])

identificateur = np.dtype([
    ('codes', 'H'),
    ('coherence', 'S9'),
    ('versionMaj', 'B'),
    ('versionMin', 'B'),
    ('versionCompil', 'H'),
    ('crypted', 'H'),
    ('nbBlocks', 'H'),
    ('firstBlockPos', 'I'),
    ('nbSeq', 'H'),
    ('recordingSeqNbr', 'H'),
    ('acqTest', 'I'),
    ('prevFileNbr', 'i'),
    ('thisFileNbr', 'I'),
    ('nextFileNbr', 'i'),
    ('thisFileName', 'B'),
])

infoPatient = np.dtype([
    ('PatientName', 'S50'),
    ('PatientFirstName', 'S30'),
    ('birthday', 'S10'),
    ('sexe', 'S1'),
    ('patientNumber', 'S20'),
    ('originalRecordingCenter', 'S39'),
    ('remark', 'S256'),
])

descripteur1024 = np.dtype([
    ('dscType', 'I'),
    ('videoShift', 'i'),
    ('recordLenInSeconds', 'I'),
    ('recordStart', '7B'),
    ('videoType', 'H'),
    ('videoDrift', 'h'),
    ('nbDataBlock', 'H'),
    ('nbFrequencyGroup', 'H'),
    ('sampleFrequencyGroup', '%dI' % (MAXGRP,)),
    ('sampleFrequencyTimeBase', '%dI' % (MAXGRP,)),
    ('totalNbChannels', 'H'),
    ('nbUsedChannels', 'H'),
    ('channelName', '%dS8' % (MAXELEC,)),
    ('channelKind', '%dB' % (MAXELEC,)),
    ('frequencyGroupNumber', '%dB' % (MAXELEC,)),
    ('theta', '%db' % (MAXELEC,)),
    ('phi', '%db' % (MAXELEC,)),
    ('rho', '%dB' % (MAXELEC,)),
    ('analogMin', '%dI' % (MAXELEC,)),
    ('analogMax', '%dI' % (MAXELEC,)),
    ('convMin', '%dI' % (MAXELEC,)),
    ('convMax', '%dI' % (MAXELEC,)),
    ('dataType', '%dB' % (4 * MAXELEC,)),
    ('channelSubKind', '%db' % (MAXELEC,)),
    ('notUsed_1', '%dB' % (MAXELEC, )),
    ('compressed', 'H'),
    ('thetaGround1', 'B'),
    ('phiGround1', 'B'),
    ('thetaGround2', 'B'),
    ('phiGround2', 'B'),
])

descripteur128 = np.dtype([
    ('dscType', 'I'),
    ('videoShift', 'i'),
    ('recordLenInSeconds', 'I'),
    ('recordStart', '7B'),
    ('videoType', 'H'),
    ('videoDrift', 'h'),
    ('nbDataBlock', 'H'),
    ('nbFrequencyGroup', 'H'),
    ('sampleFrequencyGroup', '%dI' % (MAXGRP,)),
    ('sampleFrequencyTimeBase', '%dI' % (MAXGRP,)),
    ('totalNbChannels', 'H'),
    ('nbUsedChannels', 'H'),
    ('channelName', '%dS8' % (OLDMAXELEC,)),
    ('channelKind', '%dB' % (OLDMAXELEC,)),
    ('frequencyGroupNumber', '%dB' % (OLDMAXELEC,)),
    ('theta', '%db' % (OLDMAXELEC,)),
    ('phi', '%db' % (OLDMAXELEC,)),
    ('rho', '%dB' % (OLDMAXELEC,)),
    ('analogMin', '%dI' % (OLDMAXELEC,)),
    ('analogMax', '%dI' % (OLDMAXELEC,)),
    ('convMin', '%dI' % (OLDMAXELEC,)),
    ('convMax', '%dI' % (OLDMAXELEC,)),
    ('dataType', '%dB' % (4 * OLDMAXELEC,)),
    ('channelSubKind', '%db' % (OLDMAXELEC,)),
    ('notUsed_1', '%dB' % (OLDMAXELEC, )),
    ('compressed', 'H'),
    ('thetaGround1', 'B'),
    ('phiGround1', 'B'),
    ('thetaGround2', 'B'),
    ('phiGround2', 'B'),
])

frequencyGroup = np.dtype([
    ('frequencyGroupNbr', 'H'),
    ('stockage', 'B'),
    ('checksum', 'B'),
    ('totalNbChannels', 'H'),
    ('NbUsedChannels', 'H')
])

cbmap = {
    0xCAFD0300: 'skip',
    0xCAFD0301: 'infoPatient',
    0xCAFD0302: 'infoMontage',
    0xCAFD0100: 'skip',
    0xCAFD0101: 'readData',
    0xCAFD0102: 'offset data',
    0xCAFD0103: 'skip_block_len',
    0xCAFD0104: 'skip_block_len',
    0xCAFD0105: 'skip_block_len',
    0xCAFD0200: 'skip',
    0xCAFD0201: 'read markers',
    0xCAFDCAFD: 'end',
}

def uncrypt(buff):
    _ = np.frombuffer(buff, dtype=np.int8).copy()
    _[:-1] ^= _[1:] ^ np.r_[1:len(buff)]
    return _.tostring()


class EEG(object):

    def __init__(self, filename, verbose=True):
        fd = open(filename, 'rb')
        if verbose:
            def log(msg):
                print (msg)
        else:
            log = lambda msg: msg
        # read file header
        log('reading header from %r' % fd)
        ident = np.fromfile(fd, identificateur, 1)
        if not ident['coherence'][0] == b'COHERENCE':
            raise IOError('%r is not a valid EEG file' % fd)
        # go to data
        fd.seek(ident['firstBlockPos'][0])
        while True:
            hdr = np.fromfile(fd, header, 1)
            cb = cbmap.get(hdr['codeBlock'][0])
            log('code block: %r' % cb)
            if cb == 'skip':
                continue
            elif cb == 'infoPatient':
                ip = np.fromfile(fd, infoPatient, 1)
                log('infoPatient %r' % ip)
            elif cb == 'infoMontage':
                im = np.fromfile(fd, infoMontage, 1)
                log('infoMontage %r' % im)
            elif cb == 'readData':
                file_version = ident['versionMaj'] * 10 + ident['versionMin']
                desctype = descripteur1024 if file_version >= 36 else descripteur128
                buff = uncrypt(fd.read(desctype.itemsize))
                desc = np.array(buff, dtype=desctype)
                log('data descriptor')
                for k in desc.dtype.fields.keys():
                    v = desc[k]
                    if v.size > 10:
                        log("%20s\t%15s\t%15s" % (k, v.dtype, v.shape))
                    else:
                        log("%20s\t%15s" % (k, v))
                recordStart = desc['recordStart']
                nchan = desc['nbUsedChannels']
                srate = desc['sampleFrequencyGroup'][0] * 1.0
                chnm = [c.strip() for c in desc['channelName'] if c.strip()]
                chscale = (desc['analogMax'] * 1.0 / desc['convMax'])[:nchan]
                chkind = [{0: 'eeg', 4: 'seeg'}.get(k, 'unknown')
                          for k in desc['channelKind'][:nchan]]
            elif cb == 'offset data':
                data_offset = fd.tell()
                nsamp = (
                    hdr['blockLen'][0] - frequencyGroup.itemsize) / 2 / nchan
                log('%d samples of data in file' % nsamp)
                fd.seek(hdr['blockLen'][0], 1)
            elif cb == 'skip_block_len':
                fd.seek(hdr['blockLen'][0], 1)
            elif cb == 'read markers':
                log('ignoring markers')
                continue
            elif cb is None:
                log('done reading header')
                break
        log('closing %r' % fd)
        fd.close()
        for k in locals().keys():
            setattr(self, k, locals()[k])

    def read_data(self, t0=0.0, t1=None):
        # determine sample indices
        i0 = int(t0 * self.srate)
        if t1 is None:
            t1 = self.desc['recordLenInSeconds']
        i1 = int(t1 * self.srate)
        # pos
        print('reading from sample %d to sample %s', i0, i1)
        pos = self.data_offset + frequencyGroup.itemsize
        #pos += i0 * self.nchan * 2
        # read buff
        with open(self.filename, 'rb') as fd:
            fd.seek(pos)
            #buff = fd.read((i1 - i0) * self.nchan * 2)
            buff = fd.read(int(self.nsamp) * self.nchan * 2)
        # convert to NumPy
        data = np.frombuffer(buff, np.int16).reshape((-1, self.nchan)).T
        data = data.astype(np.float32) * self.chscale[:, np.newaxis]
        t = np.r_[t0:t1:1j * data.shape[1]]
        return data.astype(np.float32), t

    def to_fif(self):
        import mne
        from mne.io.array import RawArray
        print('have mne %r, version %r', mne, mne.__version__)
        try:
            from mne.io.meas_info import create_info
        except ImportError:
            from mne.io.array import create_info
        print('reading data')
        data, _ = self.read_data()
        print('creating info')
        chnm = [_.decode('ascii') for _ in self.chnm]
        info = create_info(chnm, self.srate, ['seeg' for _ in data])
        print('creating raw')
        raw = RawArray(data * 1e-6, info)
        return raw

    def to_mat(self, name):
        import scipy.io
        data, time = self.read_data()
        scipy.io.savemat(name, {'data': data, 'time': time, 'channel_names': self.chnm, 'sampling_rate_hz': self.srate})

    def to_npy(self):
        return self.read_data()

    def save_to_fif(self, filename):
        raw = self.to_fif()
        print('saving %r to %r', raw, filename)
        raw.save(filename)
        print('done')

    def save_to_npy(self, filename):
        raw = self.to_npy()
        print('saving %r to %r', raw, filename)
        np.save(filename, raw)
        print('done')


if __name__ == '__main__':
    import sys
    for fname in sys.argv[1:]:
        print ('converting %r..' % (fname, ))
        try:
            EEG(fname).save_to_fif(fname + '-raw.fif')
        except Exception as exc:
            print('unable to convert: %r' % (exc, ))
            
