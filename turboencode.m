function codebits=turboencode(msgint,intrlvrIndices)
hTEnc = comm.TurboEncoder('TrellisStructure',poly2trellis(4, ...
    [13 15 17],13),'InterleaverIndices',intrlvrIndices);
codebits=step(hTEnc,msgint);
end