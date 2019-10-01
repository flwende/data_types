// Copyright (c) 2017-2019 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License
// (See accompanying file LICENSE)

#if !defined(AUXILIARY_MACRO_HPP)
#define AUXILIARY_MACRO_HPP

#define FW_CONCAT_(X_1, X_2) X_1 ## X_2
#define FW_CONCAT(X_1, X_2) FW_CONCAT_(X_1, X_2)

#define FW_SEQ_MAX_N 16

#define FW_SEQ_1(X) X
#define FW_SEQ_2(X) FW_SEQ_1(X), FW_SEQ_1(X)
#define FW_SEQ_4(X) FW_SEQ_2(X), FW_SEQ_2(X)
#define FW_SEQ_8(X) FW_SEQ_4(X), FW_SEQ_4(X)
#define FW_SEQ_16(X) FW_SEQ_8(X), FW_SEQ_8(X)
#define FW_SEQ_32(X) FW_SEQ_16(X), FW_SEQ_16(X)
#define FW_SEQ_64(X) FW_SEQ_32(X), FW_SEQ_32(X)
#define FW_SEQ_N FW_CONCAT(FW_SEQ_, FW_SEQ_MAX_N)

#endif