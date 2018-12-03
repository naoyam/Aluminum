////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.  Produced at the
// Lawrence Livermore National Laboratory in collaboration with University of
// Illinois Urbana-Champaign.
//
// Written by the LBANN Research Team (N. Dryden, N. Maruyama, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-756777.
// All rights reserved.
//
// This file is part of Aluminum GPU-aware Communication Library. For details, see
// http://software.llnl.gov/Aluminum or https://github.com/LLNL/Aluminum.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "Al.hpp"
#include "mpi_cuda/communicator.hpp"
#include "mpi_cuda/allreduce_ring.hpp"
#include "mpi_cuda/rma.hpp"

namespace Al {
namespace internal {
namespace mpi_cuda {

RingMPICUDA &MPICUDACommunicator::get_ring() {
  if (!m_ring)
    m_ring = new RingMPICUDA(*this);
  return *m_ring;
}

void MPICUDACommunicator::init_rma() {
  if (!m_rma) {
    std::cerr << "Initializing rma" << std::endl;
    m_rma = std::make_shared<RMA>(*this);
  }
}
RMA &MPICUDACommunicator::get_rma() {
  init_rma();
  return *m_rma;
}

MPICUDACommunicator::~MPICUDACommunicator() {
  if (m_ring)
    delete m_ring;
}

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
