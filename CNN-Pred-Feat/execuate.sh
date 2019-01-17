#!/bin/bash
eval "THC_CACHING_ALLOCATOR=0 th tv-conv-freture-has-subfodler.lua"
eval "mv convlstm_split1_tv-feattest30.t7 /data_yww"
eval "THC_CACHING_ALLOCATOR=0 th RGB-conv-freture-has-subfodler.lua"

