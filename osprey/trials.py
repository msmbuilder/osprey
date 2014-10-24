import json
from datetime import datetime, timedelta

from six import iteritems

from sqlalchemy import Column, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import (TypeDecorator, Text, Float, Integer, Enum,
                              DateTime, String, Interval)
from sqlalchemy.orm import Session
Base = declarative_base()

__all__ = ['Trial']


class JSONEncoded(TypeDecorator):
    impl = Text

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class Trial(Base):
    __tablename__ = 'trails'

    id = Column(Integer, primary_key=True)
    status = Column(Enum('PENDING', 'SUCCEEDED', 'FAILED'))
    parameters = Column(JSONEncoded())
    mean_cv_score = Column(Float)
    cv_scores = Column(JSONEncoded())

    started = Column(DateTime())
    completed = Column(DateTime())
    elapsed = Column(Interval())
    host = Column(String(512))
    user = Column(String(512))
    traceback = Column(Text())
    config_sha1 = Column(String(40))

    def to_dict(self):
        item = {}
        for k, v in iteritems(self.__dict__):
            if k.startswith('_'):
                continue
            if isinstance(v, (datetime, timedelta)):
                v = str(v)
            item[k] = v
        return item


def make_session(uri, table_name='trials', echo=False):
    Trial.__tablename__ = table_name
    engine = create_engine(uri, echo=echo)
    Base.metadata.create_all(engine)
    session = Session(engine)
    return session
