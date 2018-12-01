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

#pragma once

#include "mpi_cuda/rma.hpp"
#include <cstring>
#include <set>

namespace Al {
namespace internal {
namespace mpi_cuda {

class MemHandleIPC: public MemHandle {
 public:
  MemHandleIPC(AlRequest req): MemHandle(), m_req(req) {}
  virtual ~MemHandleIPC() override {}
  virtual void *get() override {
    internal::get_progress_engine()->wait_for_completion(m_req);
    return MemHandle::get();
  }
  virtual AlRequest &request() {
    return m_req;
  }
 protected:
  AlRequest m_req;
};

namespace rma_ipc {

void set_device_for_peer_copy(int peer, bool peer_access_enabled) {
  if (!peer_access_enabled) {
    AL_CHECK_CUDA(cudaSetDevice(peer));
  }
}

void reset_device(int self, bool peer_access_enabled) {
  if (!peer_access_enabled) {
    AL_CHECK_CUDA(cudaSetDevice(self));
  }
}

class ConnectState: public AlState {
 public:
  ConnectState(AlRequest req, int peer, MPICUDACommunicator &comm,
               cudaEvent_t ev, cudaEvent_t &ev_peer):
      AlState(req), m_peer(peer), m_comm(comm), m_ev(ev), m_ev_peer(ev_peer) {}
  void start() override {
    std::cerr << "Starting ConnectState from "
              << m_comm.rank() << " to " << m_peer
              << " " << m_comm.get_comm()
              << std::endl;
    AL_CHECK_CUDA(cudaIpcGetEventHandle(&m_ipc_handle_self, m_ev));
    MPI_Isend(&m_ipc_handle_self, sizeof(cudaIpcEventHandle_t),
              MPI_BYTE, m_peer, 0, m_comm.get_comm(), &m_requests[0]);
    MPI_Irecv(&m_ipc_handle_peer, sizeof(cudaIpcEventHandle_t),
              MPI_BYTE, m_peer, 0, m_comm.get_comm(), &m_requests[1]);
  }
  bool step() override {
    int flag;
    MPI_Testall(2, m_requests, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      AL_CHECK_CUDA(
          cudaIpcOpenEventHandle(&m_ev_peer, m_ipc_handle_peer));
      std::cerr << "Connected to " << m_peer << std::endl;
      return true;
    }
    return false;
  }
 private:
  int m_peer;
  MPICUDACommunicator &m_comm;
  cudaEvent_t m_ev;
  cudaEvent_t &m_ev_peer;
  MPI_Request m_requests[2];
  cudaIpcEventHandle_t m_ipc_handle_self;
  cudaIpcEventHandle_t m_ipc_handle_peer;
};

class HostSyncState: public AlState {
 public:
  HostSyncState(AlRequest req, int peer, MPICUDACommunicator &comm):
      AlState(req), m_peer(peer), m_comm(comm) {}
  void start() override {
    MPI_Isend(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_requests[0]);
    MPI_Irecv(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_requests[1]);
  }
  bool step() override {
    int flag;
    MPI_Testall(2, m_requests, &flag, MPI_STATUS_IGNORE);
    return flag;
  }
 private:
  int m_peer;
  MPICUDACommunicator &m_comm;
  int key = 0;
  MPI_Request m_requests[2];
};

class AttachState: public AlState {
 public:
  AttachState(AlRequest req, int peer,
              int dev, int dev_peer,
              MPICUDACommunicator &comm,
              void *local_addr, MPICUDABackend::mem_handle_type peer_handle,
              bool peer_access_enabled):
      AlState(req), m_peer(peer), m_dev(dev), m_dev_peer(dev_peer), m_comm(comm),
      m_local_addr(local_addr), m_peer_handle(peer_handle),
      m_peer_access_enabled(peer_access_enabled) {}
  void start() override {
    if (m_local_addr != nullptr) {
      AL_CHECK_CUDA(cudaIpcGetMemHandle(&m_ipc_handle_self, m_local_addr));
    } else {
      // Clears the handle if the local pointer is null
      std::memset(&m_ipc_handle_self, 0, sizeof(cudaIpcMemHandle_t));
    }
    MPI_Isend(&m_ipc_handle_self, sizeof(cudaIpcMemHandle_t), MPI_BYTE, m_peer, 0,
              m_comm.get_comm(), &m_requests[0]);
    MPI_Irecv(&m_ipc_handle_peer, sizeof(cudaIpcMemHandle_t), MPI_BYTE, m_peer, 0,
              m_comm.get_comm(), &m_requests[1]);
  }
  bool step() override {
    int flag;
    MPI_Testall(2, m_requests, &flag, MPI_STATUS_IGNORE);
    if (!flag) return false;
    cudaIpcMemHandle_t void_handle;
    std::memset(&void_handle, 0, sizeof(cudaIpcMemHandle_t));
    // Remote address is not a valid address
    if (std::memcmp(&m_ipc_handle_peer, &void_handle,
                    sizeof(cudaIpcMemHandle_t)) == 0) {
      return true;
    }
    set_device_for_peer_copy(m_dev_peer, m_peer_access_enabled);
    void *peer_addr = nullptr;
#if 0
    AL_CHECK_CUDA(
        cudaIpcOpenMemHandle(&peer_addr, m_ipc_handle_peer,
                             cudaIpcMemLazyEnablePeerAccess));
#else
    {
      auto e = cudaIpcOpenMemHandle(&peer_addr, m_ipc_handle_peer,
                                    cudaIpcMemLazyEnablePeerAccess);
      if (e != cudaSuccess) {
        std::cerr << "IpcOpenMemHandle failed: "
                  << "m_dev: " << m_dev << ", m_dev_peer: " << m_dev_peer
                  << ", enabled: " << m_peer_access_enabled << std::endl;
        std::abort();
      }
    }
#endif
    reset_device(m_dev, m_peer_access_enabled);
    m_peer_handle->set(peer_addr);
    return true;
  }
 private:
  int m_peer;
  int m_dev;
  int m_dev_peer;
  MPICUDACommunicator &m_comm;
  void *m_local_addr;
  MPICUDABackend::mem_handle_type m_peer_handle;
  bool m_peer_access_enabled;
  MPI_Request m_requests[2];
  cudaIpcMemHandle_t m_ipc_handle_self;
  cudaIpcMemHandle_t m_ipc_handle_peer;
};

class NotifyState: public AlState {
 public:
  NotifyState(AlRequest req, int peer, MPICUDACommunicator &comm,
              cudaEvent_t ev):
      AlState(req), m_peer(peer), m_comm(comm), m_ev(ev) {}
  void start() override {
    AL_CHECK_CUDA(cudaEventRecord(m_ev, m_comm.get_stream()));
    MPI_Isend(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_requests[0]);
    MPI_Irecv(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_requests[1]);
  }
  bool step() override {
    int flag;
    MPI_Testall(2, m_requests, &flag, MPI_STATUS_IGNORE);
    return flag;
  }
 private:
  int key = 0;
  int m_peer;
  MPICUDACommunicator &m_comm;
  cudaEvent_t m_ev;
  MPI_Request m_requests[2];
};

class WaitState: public AlState {
 public:
  WaitState(AlRequest req, int peer, MPICUDACommunicator &comm,
            cudaEvent_t ev_peer):
      AlState(req), m_peer(peer), m_comm(comm),
      m_ev_peer(ev_peer),
      m_stream_wait_set(false) {}
  void start() override {
    MPI_Irecv(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_req);
  }
  bool step() override {
    int flag;
    MPI_Test(&m_req, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      if (!m_stream_wait_set) {
        AL_CHECK_CUDA(cudaStreamWaitEvent(
            m_comm.get_stream(), m_ev_peer, 0));
        MPI_Isend(&key, 1, MPI_INT, m_peer, 0,
                  m_comm.get_comm(), &m_req);
        m_stream_wait_set = true;
        return false;
      } else {
        return true;
      }
    }
    return false;
  }
 private:
  int key = 0;
  int m_peer;
  MPICUDACommunicator &m_comm;
  cudaEvent_t m_ev_peer;
  bool m_stream_wait_set;
  MPI_Request m_req;
};

class SyncState: public AlState {
 public:
  SyncState(AlRequest req, int peer, MPICUDACommunicator &comm,
            cudaEvent_t ev_self, cudaEvent_t ev_peer):
      AlState(req), m_peer(peer), m_comm(comm),
      m_ev_self(ev_self), m_ev_peer(ev_peer),
      m_stream_wait_set(false) {}
  void start() override {
    AL_CHECK_CUDA(cudaEventRecord(m_ev_self, m_comm.get_stream()));
    MPI_Isend(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_requests[0]);
    MPI_Irecv(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_requests[1]);
  }
  bool step() override {
    int flag;
    MPI_Testall(2, m_requests, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      if (!m_stream_wait_set) {
        AL_CHECK_CUDA(cudaStreamWaitEvent(
            m_comm.get_stream(), m_ev_peer, 0));
        MPI_Isend(&key, 1, MPI_INT, m_peer, 0,
                  m_comm.get_comm(), &m_requests[0]);
        MPI_Irecv(&key, 1, MPI_INT, m_peer, 0,
                  m_comm.get_comm(), &m_requests[1]);
        m_stream_wait_set = true;
      } else {
        return true;
      }
    }
    return false;
  }
 private:
  int key = 0;
  int m_peer;
  MPICUDACommunicator &m_comm;
  cudaEvent_t m_ev_self;
  cudaEvent_t m_ev_peer;
  bool m_stream_wait_set;
  MPI_Request m_requests[2];
};

}

class ConnectionIPC: public Connection {
 public:
  ConnectionIPC(MPICUDACommunicator &comm, int peer, int dev):
      Connection(comm, peer), m_dev_peer(dev),
      m_peer_access_enabled(false), m_connected(false),
      m_conn_req(NULL_REQUEST),
      m_sync_req(NULL_REQUEST) {
    AL_CHECK_CUDA(cudaGetDevice(&m_dev));
    try_enable_peer_access();
    AL_CHECK_CUDA(cudaEventCreateWithFlags(
        &m_ev, cudaEventInterprocess | cudaEventDisableTiming));
  }

  ~ConnectionIPC() override {
    AlRequest req = internal::get_free_request();
    disconnect(req);
    internal::get_progress_engine()->wait_for_completion(req);
    AL_CHECK_CUDA(cudaEventDestroy(m_ev));
  }

  void connect() override {
    if (m_connected) return;
    m_conn_req = get_free_request();
    rma_ipc::ConnectState* state =
        new rma_ipc::ConnectState(m_conn_req, m_peer, m_comm, m_ev,
                                  m_ev_peer);
    internal::get_progress_engine()->enqueue(state);
    m_connected = true;
  }

  void disconnect(AlRequest &req) override {
    if (!m_connected) {
      req->store(true, std::memory_order_release);
      return;
    }
    std::cerr << "Closing IPC conn: m_dev: " << m_dev << " to " << m_dev_peer
              << std::endl;
    detach_all_remote_buffers();
    internal::get_progress_engine()->wait_for_completion(m_sync_req);
    AL_CHECK_CUDA(cudaEventDestroy(m_ev_peer));
    rma_ipc::HostSyncState* state =
        new rma_ipc::HostSyncState(req, m_peer, m_comm);
    internal::get_progress_engine()->enqueue(state);
    m_connected = false;
  }

  bool is_connected() const override {
    return m_connected;
  }

  std::shared_ptr<MemHandle> attach_remote_buffer(void *local_addr) override {
    connect();
    internal::get_progress_engine()->wait_for_completion(m_sync_req);
    AlRequest req = internal::get_free_request();
    auto handle = std::make_shared<MemHandleIPC>(req);
    rma_ipc::AttachState* state =
        new rma_ipc::AttachState(req, m_peer, m_dev, m_dev_peer, m_comm,
                                 local_addr, handle, m_peer_access_enabled);
    internal::get_progress_engine()->enqueue(state);
    m_remote_buffers.insert(handle);
    rma_ipc::set_device_for_peer_copy(m_dev_peer, m_peer_access_enabled);
    rma_ipc::reset_device(m_dev, m_peer_access_enabled);
    return handle;
  }

  void detach_remote_buffer(MPICUDABackend::mem_handle_type mem_handle) override {
    if (!m_connected) {
      throw_al_exception("Not connected");
    }
    auto it = m_remote_buffers.find(mem_handle);
    if (it == m_remote_buffers.end()) {
      throw_al_exception("Invalid remote memory handle");
    }
    internal::get_progress_engine()->wait_for_completion(m_sync_req);
    std::shared_ptr<MemHandleIPC> ipch =
        std::dynamic_pointer_cast<MemHandleIPC>(mem_handle);
    internal::get_progress_engine()->wait_for_completion(ipch->request());
    if (mem_handle->get()) {
      rma_ipc::set_device_for_peer_copy(m_dev_peer, m_peer_access_enabled);
      AL_CHECK_CUDA(cudaIpcCloseMemHandle(mem_handle->get()));
      rma_ipc::reset_device(m_dev, m_peer_access_enabled);
    }
    m_remote_buffers.erase(it);
  }

  void detach_all_remote_buffers() override {
    for (auto &x: m_remote_buffers) {
      detach_remote_buffer(x);
    }
  }

  void notify(AlRequest &req) {
    if (!m_connected) {
      throw_al_exception("Not connected");
    }
    rma_ipc::NotifyState* state =
        new rma_ipc::NotifyState(req, m_peer, m_comm, m_ev);
    internal::get_progress_engine()->enqueue(state);
  }

  void wait(AlRequest &req) {
    if (!m_connected) {
      throw_al_exception("Not connected");
    }
    rma_ipc::WaitState* state =
        new rma_ipc::WaitState(req, m_peer, m_comm, m_ev_peer);
    internal::get_progress_engine()->enqueue(state);
  }

  void sync() {
    if (!m_connected) {
      std::stringstream msg;
      msg << "Not connected to " << m_peer;
      throw_al_exception(msg.str());
    }
    internal::get_progress_engine()->wait_for_completion(m_conn_req);
    internal::get_progress_engine()->wait_for_completion(m_sync_req);
    m_sync_req = get_free_request();
    rma_ipc::SyncState* state =
        new rma_ipc::SyncState(m_sync_req, m_peer, m_comm, m_ev, m_ev_peer);
    internal::get_progress_engine()->enqueue(state);
  }

  void put(const void *src, MPICUDABackend::mem_handle_type dst, size_t size)
      override {
    if (size > 0) {
      if (!m_connected) {
        throw_al_exception("Not connected");
      }
      internal::get_progress_engine()->wait_for_completion(m_conn_req);
      internal::get_progress_engine()->wait_for_completion(m_sync_req);
      if (src == nullptr) {
        throw_al_exception("Source buffer is null");
      }
      if (dst == nullptr) {
        throw_al_exception("Destination buffer is null");
      }
      AL_CHECK_CUDA(cudaMemcpyPeerAsync(dst->get(), m_dev_peer, src, m_dev,
                                        size, m_comm.get_stream()));
    }
  }

 private:
  int m_dev;
  int m_dev_peer;
  bool m_peer_access_enabled;
  bool m_connected;
  cudaEvent_t m_ev;
  cudaEvent_t m_ev_peer;
  std::set<MPICUDABackend::mem_handle_type> m_remote_buffers;
  AlRequest m_conn_req;
  AlRequest m_sync_req;

  void try_enable_peer_access() {
    if (!m_peer_access_enabled) {
      int peer_access;
      AL_CHECK_CUDA(
          cudaDeviceCanAccessPeer(&peer_access, m_dev, m_dev_peer));
      cudaError_t e = cudaDeviceEnablePeerAccess(m_dev_peer, 0);
      if (!(e == cudaSuccess ||
            e == cudaErrorPeerAccessAlreadyEnabled)) {
        throw_al_exception("Enabling peer access failed");
      }
      m_peer_access_enabled = true;
      // clear the error status
      cudaGetLastError();
    }
  }

};

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
