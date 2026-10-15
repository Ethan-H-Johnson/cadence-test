from glob import glob

from setuptools import setup

package = "limo_cadence_bridge"
setup(
    name=package,
    version="0.1.0",
    packages=[package],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package]),
        ("share/" + package, ["package.xml"]),
        ("share/" + package + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ethan Johnson",
    maintainer_email="ehayes4560@gmail.com",
    description="TCP bridge between a Cadence controller and the LIMO's ROS 2 topics.",
    license="GPL-3.0-or-later",
    entry_points={"console_scripts": ["bridge = limo_cadence_bridge.bridge:main"]},
)
