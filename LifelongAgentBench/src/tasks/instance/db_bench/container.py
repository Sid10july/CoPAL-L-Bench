import docker
import mysql.connector
import random
import socket
import time
import os
from docker.models import containers
from typing import Optional


class DBBenchContainer:
    port = 13000
    password = "password"

    def __init__(self, image: str = "mysql"):
        self.deleted = False
        self.image = image
        self.client = None
        self.container = None
        self.conn = None

        # External MySQL mode: use sidecar MySQL in the same pod.
        host = os.getenv("DBBENCH_MYSQL_HOST")
        port_env = os.getenv("DBBENCH_MYSQL_PORT")
        self.password = os.getenv("DBBENCH_MYSQL_PASSWORD", self.password)
        if host and port_env:
            self.host = host
            self.port = int(port_env)
            self._connect()
            return

        # Docker mode: original behavior for environments with Docker daemon.
        self.client = docker.from_env()
        p = DBBenchContainer.port + random.randint(0, 10000)
        while self.is_port_open(p):
            p += random.randint(0, 20)
        self.port = p
        self.container: containers.Container = self.client.containers.run(
            image,
            name=f"mysql_{self.port}",
            environment={"MYSQL_ROOT_PASSWORD": self.password},
            ports={"3306": self.port},
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
        )

        time.sleep(1)
        self.host = "127.0.0.1"
        self._connect()

    def _connect(self) -> None:
        retry = 0
        while True:
            try:
                self.conn = mysql.connector.connect(
                    host=self.host,
                    user="root",
                    password=self.password,
                    port=self.port,
                    pool_reset_session=True,
                )
            except (
                mysql.connector.errors.OperationalError,
                mysql.connector.errors.InterfaceError,
                mysql.connector.errors.DatabaseError,
            ):
                if retry > 60:
                    raise
                time.sleep(2)
            else:
                break
            retry += 1

    def delete(self) -> None:
        if self.container is not None:
            self.container.stop()
        self.deleted = True

    def __del__(self) -> None:
        try:
            if not self.deleted:
                self.delete()
        except Exception:  # noqa
            pass

    def execute(
        self,
        multiple_sql: str,
        database: Optional[str] = None,
    ) -> str:
        self.conn.reconnect()
        try:
            cursor = self.conn.cursor()
            if database:
                cursor.execute(f"use `{database}`;")
                cursor.fetchall()
            sql_list = multiple_sql.split(";")
            sql_list = [sql.strip() for sql in sql_list if sql.strip() != ""]
            result = ""
            for sql in sql_list:
                cursor.execute(sql)
                result = str(cursor.fetchall())
                self.conn.commit()
        except Exception as e:
            result = str(e)
        return result

    def is_port_open(
        self, port: int
    ) -> bool:  # noqa (The quality checker of the IDE is wrong)
        try:
            if self.client is not None:
                self.client.containers.get(f"mysql_{port}")
                return True
        except Exception:  # noqa
            pass

        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # use IPv4 and TCP
        try:
            # Try to connect to the specified port
            sock.connect(("localhost", port))
            # If the connection succeeds, the port is occupied
            return True
        except ConnectionRefusedError:
            # If the connection is refused, the port is not occupied
            return False
        finally:
            # Close the socket
            sock.close()
