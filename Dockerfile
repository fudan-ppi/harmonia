FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt-get clean
RUN apt-get update

RUN apt-get install -y build-essential
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cmake
# RUN apt-get install -y gdb
RUN apt-get install -y net-tools
RUN apt-get install -y vim git


RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN sed -ri 's/^#PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN sed -ri 's/#AddressFamily any/AddressFamily inet/g' /etc/ssh/sshd_config
RUN sed -ri 's/#AllowTcpForwarding yes/AllowTcpForwarding yes/g' /etc/ssh/sshd_config
RUN sed -ri 's/#X11DisplayOffset 10/X11DisplayOffset 10/g' /etc/ssh/sshd_config
RUN sed -ri 's/#X11UseLocalhost yes/X11UseLocalhost yes/g' /etc/ssh/sshd_config
RUN service ssh restart

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y xorg
RUN touch /root/.Xauthority
RUN xauth add localhost/unix:10 MIT-MAGIC-COOKIE-1 $(mcookie)
RUN service ssh restart

RUN apt-get install -y build-essential strace gdb sudo
RUN apt-get install -y python3.7
RUN apt-get install -y libboost-all-dev

WORKDIR /
RUN git clone https://bitbucket.org/icl/papi.git
WORKDIR /papi
RUN git reset --hard 606f4f6de03
WORKDIR /papi/src
RUN ./configure
RUN make -j
RUN make install

RUN echo 'root:root' | chpasswd
RUN sed -ri '$a export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH' /root/.bashrc

RUN rm -rf /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python3
RUN rm -rf /usr/bin/python
RUN ln -s /usr/bin/python3.7 /usr/bin/python
RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN python -m pip install --upgrade pip
RUN pip install cython
RUN pip install numpy
RUN pip install pathlib
RUN apt-get install -y bc

RUN mkdir -p /harmonia/open-harmonia
COPY ./ /harmonia/
WORKDIR /harmonia/open-harmonia
RUN sh build.sh
WORKDIR /harmonia/open-harmonia
RUN sh generate.sh

COPY entrypoint.sh /sbin
RUN chmod +x /sbin/entrypoint.sh
ENTRYPOINT [ "/sbin/entrypoint.sh" ]
