from wtforms import Form, StringField, SelectField

class SearchForm(Form):
    search = StringField('')