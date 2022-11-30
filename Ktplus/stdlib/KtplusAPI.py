# ****************************************************************
# Copyright (c) 2022 KotlinPlus Development Team
# Copyright (c) 2019 David Callanan
# ****************************************************************

#######################################
# Imports
#######################################

from strings_with_arrows import *

import string
import os
import sys
import math
import random

#######################################
# Consts
#######################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters + '_'
LETTERS_DIGITS = LETTERS + DIGITS

#######################################
# Errors
#######################################

class Error:
  def __init__(self, pos_start, pos_end, details, error_name=None):
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.error_name = self.__class__.__name__ if error_name is None else error_name
    self.details = details
  
  def as_string(self):
    result = f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}\n'
    result += string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end) + '\n'
    result += f'{self.error_name}: {self.details}'
    return result

class IllegalCharError(Error):
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, details)

class ExpectedCharError(Error):
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, details)

class InvalidSyntaxError(Error):
  def __init__(self, pos_start, pos_end, details=''):
    super().__init__(pos_start, pos_end, details, "SyntaxError")

class RTError(Error):
  def __init__(self, pos_start, pos_end, details, context, name=None):
    super().__init__(pos_start, pos_end, details, "RuntimeError")
    if name is not None: self.error_name = name
    self.context = context
    self.string_form = f"{self.error_name}: {self.details}"

  def as_string(self):
    result  = self.generate_traceback()
    result += f'{self.error_name}: {self.details}'
    result += '\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result

  def generate_traceback(self):
    result = ''
    pos = self.pos_start
    ctx = self.context

    while ctx:
      result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
      pos = ctx.parent_entry_pos
      ctx = ctx.parent

    return 'Traceback (most recent call last):\n' + result

class Ktplus_NameError(RTError):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, details, context, "UndefVarError")

class Ktplus_ZeroDivError(RTError):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, details, context, "ZeroDivisionError")
  
class Ktplus_DtypeError(RTError):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, details, context, "TypeError")

class Ktplus_UsageError(RTError):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, details, context, "UsageError")

class Ktplus_ValueError(RTError):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, details, context, "ValueError")

class Ktplus_FileNotFoundError(RTError):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, details, context, "FileNotFoundError")

class Ktplus_AssertError(RTError):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, details, context, "AssertionError")

class Ktplus_IdxError(RTError):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, details, context, "IndexError")

#######################################
# Position
#######################################

class Position:
  def __init__(self, idx, ln, col, fn, ftxt):
    self.idx = idx
    self.ln = ln
    self.col = col
    self.fn = fn
    self.ftxt = ftxt

  def advance(self, current_char=None):
    self.idx += 1
    self.col += 1

    if current_char == '\n':
      self.ln += 1
      self.col = 0

    return self

  def copy(self):
    return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#######################################
# Tokens
#######################################

TT_INT				= 'INT' # 1234567890
TT_FLOAT    	= 'FLOAT' # 0.123456789
TT_STRING			= 'STRING' # "abc"
TT_IDENTIFIER	= 'IDENTIFIER' # abc
TT_KEYWORD		= 'KEYWORD' # for
TT_PLUS     	= 'PLUS' # +
TT_MINUS    	= 'MINUS' # -
TT_MUL      	= 'MUL' # *
TT_DIV      	= 'DIV' # /
TT_POW				= 'POW' # ^
TT_LSH        = 'LSH' # <<
TT_RSH        = 'RSH' # >>
TT_EQ					= 'EQ' # =
TT_LPAREN   	= 'LPAREN' # (
TT_RPAREN   	= 'RPAREN' # )
TT_LSQUARE    = 'LSQUARE' # [
TT_RSQUARE    = 'RSQUARE' # ]
TT_LBRACE     = 'LBRACE' # {
TT_RBRACE     = 'RBRACE' # }
TT_EE					= 'EE' # ==
TT_NE					= 'NE' # !=
TT_LT					= 'LT' # <
TT_GT					= 'GT' # >
TT_LTE				= 'LTE' # <=
TT_GTE				= 'GTE' # >=
TT_COMMA			= 'COMMA' # ,
TT_ARROW			= 'ARROW' # ->
TT_NEWLINE		= 'NEWLINE' # \n
TT_EOF				= 'EOF' # <<EOF>>

KEYWORDS = [
  'and', # and Operation
  'or', # or Operation
  'not', # not Operation
  'if', # Condition
  'elsif', ##
  'else', ##
  'for', # For-loop
  'to', ##
  'by', ##
  'while', # While-loop
  'fun', # Function definition
  'return', # Return-loop 
  'continue', # Continue-loop
  'break', # Break-loop
  'from', # For-loop
  'import', # Import statement
  'throw', # Error Raising
  'assert', # Assert
  'try', # Try-Catch-Finally loop
  'catch',##
  'finally', ##
]

class Token:
  def __init__(self, type_, value=None, pos_start=None, pos_end=None):
    self.type = type_
    self.value = value

    if pos_start:
      self.pos_start = pos_start.copy()
      self.pos_end = pos_start.copy()
      self.pos_end.advance()

    if pos_end:
      self.pos_end = pos_end.copy()

  def matches(self, type_, value):
    return self.type == type_ and self.value == value
  
  def __repr__(self):
    if self.value: return f'{self.type}:{self.value}'
    return f'{self.type}'

#######################################
# Scanner
#######################################

class Scanner:
  def __init__(self, fn, text):
    self.fn = fn
    self.text = text
    self.pos = Position(-1, 0, -1, fn, text)
    self.current_char = None
    self.advance()
  
  def advance(self):
    self.pos.advance(self.current_char)
    self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

  def make_tokens(self):
    tokens = []

    while self.current_char != None:
      if self.current_char in ' \t':
        self.advance()
      elif self.current_char == '#':
        self.skip_comment()
      elif self.current_char in ';\n':
        tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
        self.advance()
      elif self.current_char in DIGITS:
        tokens.append(self.make_number())
      elif self.current_char in LETTERS:
        tokens.append(self.make_identifier())
      elif self.current_char == '"':
        tokens.append(self.make_string())
      elif self.current_char == '+':
        tokens.append(Token(TT_PLUS, pos_start=self.pos))
        self.advance()
      elif self.current_char == '-':
        tokens.append(self.make_minus_or_arrow())
      elif self.current_char == '*':
        tokens.append(Token(TT_MUL, pos_start=self.pos))
        self.advance()
      elif self.current_char == '/':
        tokens.append(Token(TT_DIV, pos_start=self.pos))
        self.advance()
      elif self.current_char == '^':
        tokens.append(Token(TT_POW, pos_start=self.pos))
        self.advance()
      elif self.current_char == '(':
        tokens.append(Token(TT_LPAREN, pos_start=self.pos))
        self.advance()
      elif self.current_char == ')':
        tokens.append(Token(TT_RPAREN, pos_start=self.pos))
        self.advance()
      elif self.current_char == '[':
        tokens.append(Token(TT_LSQUARE, pos_start=self.pos))
        self.advance()
      elif self.current_char == ']':
        tokens.append(Token(TT_RSQUARE, pos_start=self.pos))
        self.advance()
      elif self.current_char == '{':
        tokens.append(Token(TT_LBRACE, pos_start=self.pos))
        self.advance()
      elif self.current_char == '}':
        tokens.append(Token(TT_RBRACE, pos_start=self.pos))
        self.advance()
      elif self.current_char == '!':
        token, error = self.make_not_equals()
        if error: return [], error
        tokens.append(token)
      elif self.current_char == '=':
        tokens.append(self.make_equals())
      elif self.current_char == '<':
        tokens.append(self.make_less_than())
      elif self.current_char == '>':
        tokens.append(self.make_greater_than())
      elif self.current_char == ',':
        tokens.append(Token(TT_COMMA, pos_start=self.pos))
        self.advance()
      else:
        pos_start = self.pos.copy()
        char = self.current_char
        self.advance()
        return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

    tokens.append(Token(TT_EOF, pos_start=self.pos))
    return tokens, None

  def make_number(self):
    num_str = ''
    dot_count = 0
    pos_start = self.pos.copy()

    while self.current_char != None and self.current_char in DIGITS + '.':
      if self.current_char == '.':
        if dot_count == 1: break
        dot_count += 1
      num_str += self.current_char
      self.advance()

    if dot_count == 0:
      return Token(TT_INT, int(num_str), pos_start, self.pos)
    else:
      return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

  def make_string(self):
    string = ''
    pos_start = self.pos.copy()
    escape_character = False
    self.advance()

    escape_characters = {
      'n': '\n',
      't': '\t',
      '"': '"'
    }

    while self.current_char != None and (self.current_char != '"' or escape_character):
      if escape_character:
        string += escape_characters.get(self.current_char, self.current_char)
      else:
        if self.current_char == '\\':
          escape_character = True
          self.advance()
          continue
        else:
          string += self.current_char
      self.advance()
      escape_character = False

    self.advance()
    return Token(TT_STRING, string, pos_start, self.pos)

  def make_identifier(self):
    id_str = ''
    pos_start = self.pos.copy()

    while self.current_char != None and self.current_char in LETTERS_DIGITS:
      id_str += self.current_char
      self.advance()

    tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
    return Token(tok_type, id_str, pos_start, self.pos)

  def make_minus_or_arrow(self):
    tok_type = TT_MINUS
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '>':
      self.advance()
      tok_type = TT_ARROW

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_not_equals(self):
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None

    self.advance()
    return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")
  
  def make_equals(self):
    tok_type = TT_EQ
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_EE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_less_than(self):
    tok_type = TT_LT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_LTE
    elif self.current_char == '<':
      self.advance()
      tok_type = TT_LSH

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_greater_than(self):
    tok_type = TT_GT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_GTE
    elif self.current_char == '>':
      self.advance()
      tok_type = TT_RSH

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def skip_comment(self):
    self.advance()

    while self.current_char != '\n' and self.current_char != ';':
      self.advance()

    self.advance()

#######################################
# Nodes
#######################################
 
class NumberNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class StringNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class ListNode:
  def __init__(self, element_nodes, pos_start, pos_end):
    self.element_nodes = element_nodes

    self.pos_start = pos_start
    self.pos_end = pos_end

class VarAccessNode:
  def __init__(self, var_name_tok):
    self.var_name_tok = var_name_tok

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
  def __init__(self, var_name_tok, value_node):
    self.var_name_tok = var_name_tok
    self.value_node = value_node

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.value_node.pos_end

class BinOpNode:
  def __init__(self, left_node, op_tok, right_node):
    self.left_node = left_node
    self.op_tok = op_tok
    self.right_node = right_node

    self.pos_start = self.left_node.pos_start
    self.pos_end = self.right_node.pos_end

  def __repr__(self):
    return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
  def __init__(self, op_tok, node):
    self.op_tok = op_tok
    self.node = node

    self.pos_start = self.op_tok.pos_start
    self.pos_end = node.pos_end

  def __repr__(self):
    return f'({self.op_tok}, {self.node})'

class IfNode:
  def __init__(self, cases, else_case):
    self.cases = cases
    self.else_case = else_case

    self.pos_start = self.cases[0][0].pos_start
    self.pos_end = (self.else_case or self.cases[len(self.cases) - 1])[0].pos_end

class TryNode:
  def __init__(self, cases, finally_case):
    self.finally_case = finally_case
    self.cases = cases

    self.pos_start = self.cases[0][0].pos_start
    self.pos_end = (self.finally_case or self.cases[len(self.cases) - 1])[0].pos_end

class ForNode:
  def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
    self.var_name_tok = var_name_tok
    self.start_value_node = start_value_node
    self.end_value_node = end_value_node
    self.step_value_node = step_value_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.body_node.pos_end

class ImportNode:
  def __init__(self, tok):
    self.tok = tok
    
    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

class WhileNode:
  def __init__(self, condition_node, body_node, should_return_null):
    self.condition_node = condition_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.pos_start = self.condition_node.pos_start
    self.pos_end = self.body_node.pos_end

class FuncDefNode:
  def __init__(self, var_name_tok, arg_name_toks, body_node, should_auto_return):
    self.var_name_tok = var_name_tok
    self.arg_name_toks = arg_name_toks
    self.body_node = body_node
    self.should_auto_return = should_auto_return

    if self.var_name_tok:
      self.pos_start = self.var_name_tok.pos_start
    elif len(self.arg_name_toks) > 0:
      self.pos_start = self.arg_name_toks[0].pos_start
    else:
      self.pos_start = self.body_node.pos_start

    self.pos_end = self.body_node.pos_end

class CallNode:
  def __init__(self, node_to_call, arg_nodes):
    self.node_to_call = node_to_call
    self.arg_nodes = arg_nodes

    self.pos_start = self.node_to_call.pos_start

    if len(self.arg_nodes) > 0:
      self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
    else:
      self.pos_end = self.node_to_call.pos_end

class ReturnNode:
  def __init__(self, node_to_return, pos_start, pos_end):
    self.node_to_return = node_to_return

    self.pos_start = pos_start
    self.pos_end = pos_end

class ContinueNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

class BreakNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

class AssertNode:
  def __init__(self, condition, pos_start, pos_end):
    self.condition_node = condition

    self.pos_start = self.condition_node.pos_start
    self.pos_end = self.condition_node.pos_end

class ThrowNode:
  def __init__(self, details_tok, pos_start, pos_end):
    self.details_tok = details_tok

    self.pos_start = pos_start
    self.pos_end = pos_end

#######################################
# Parse Result
#######################################

class ParseResult:
  def __init__(self):
    self.error = None
    self.node = None
    self.last_registered_advance_count = 0
    self.advance_count = 0
    self.to_reverse_count = 0

  def register_advancement(self):
    self.last_registered_advance_count = 1
    self.advance_count += 1

  def register(self, res):
    self.last_registered_advance_count = res.advance_count
    self.advance_count += res.advance_count
    if res.error: self.error = res.error
    return res.node

  def try_register(self, res):
    if res.error:
      self.to_reverse_count = res.advance_count
      return None
    return self.register(res)

  def success(self, node):
    self.node = node
    return self

  def failure(self, error):
    if not self.error or self.last_registered_advance_count == 0:
      self.error = error
    return self

#######################################
# Parser
#######################################

class Parser:
  def __init__(self, tokens):
    self.tokens = tokens
    self.tok_idx = -1
    self.advance()

  def advance(self):
    self.tok_idx += 1
    self.update_current_tok()
    return self.current_tok

  def reverse(self, amount=1):
    self.tok_idx -= amount
    self.update_current_tok()
    return self.current_tok

  def update_current_tok(self):
    if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
      self.current_tok = self.tokens[self.tok_idx]

  def parse(self):
    res = self.statements()
    if not res.error and self.current_tok.type != TT_EOF:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Token cannot appear after previous tokens"
      ))
    return res

  ###################################

  def statements(self):
    res = ParseResult()
    statements = []
    pos_start = self.current_tok.pos_start.copy()

    while self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

    statement = res.register(self.statement())
    if res.error: return res
    statements.append(statement)

    more_statements = True

    while True:
      newline_count = 0
      while self.current_tok.type == TT_NEWLINE:
        res.register_advancement()
        self.advance()
        newline_count += 1
      if newline_count == 0:
        more_statements = False
      
      if not more_statements: break
      statement = res.try_register(self.statement())
      if not statement:
        self.reverse(res.to_reverse_count)
        more_statements = False
        continue
      statements.append(statement)

    return res.success(ListNode(
      statements,
      pos_start,
      self.current_tok.pos_end.copy()
    ))

  def statement(self):
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.matches(TT_KEYWORD, 'return'):
      res.register_advancement()
      self.advance()

      expr = res.try_register(self.expr())
      if not expr:
        self.reverse(res.to_reverse_count)
      return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TT_KEYWORD, 'continue'):
      res.register_advancement()
      self.advance()
      return res.success(ContinueNode(pos_start, self.current_tok.pos_start.copy()))
      
    if self.current_tok.matches(TT_KEYWORD, 'break'):
      res.register_advancement()
      self.advance()
      return res.success(BreakNode(pos_start, self.current_tok.pos_start.copy()))

    expr = res.register(self.expr())
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'return', 'continue', 'break', 'var', 'if', 'for', 'while', 'fun', int, float, identifier, '+', '-', '(', '[' or 'not'"
      ))
    return res.success(expr)

  def expr(self):
    res = ParseResult()

    if self.current_tok.type == TT_IDENTIFIER and self.tokens[self.tok_idx+1].type == TT_EQ:
      var_name = self.current_tok
      res.register_advancement()
      self.advance()

      if self.current_tok.type != TT_EQ:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected '='"
        ))

      res.register_advancement()
      self.advance()
      expr = res.register(self.expr())
      if res.error: return res
      return res.success(VarAssignNode(var_name, expr))

    node = res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or'))))

    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'var', 'if', 'for', 'while', 'fun', int, float, identifier, '+', '-', '(', '[' or 'not'"
      ))

    return res.success(node)

  def comp_expr(self):
    res = ParseResult()

    if self.current_tok.matches(TT_KEYWORD, 'not'):
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()

      node = res.register(self.comp_expr())
      if res.error: return res
      return res.success(UnaryOpNode(op_tok, node))
    
    node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))
    
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected int, float, identifier, '+', '-', '(', '[', 'if', 'for', 'while', 'fun' or 'not'"
      ))

    return res.success(node)
  def arith_expr(self):
    return self.bin_op(self.arith_pm_expr, (TT_LSH, TT_RSH))

  def arith_pm_expr(self):
    return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

  def term(self):
    return self.bin_op(self.factor, (TT_MUL, TT_DIV))

  def factor(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TT_PLUS, TT_MINUS):
      res.register_advancement()
      self.advance()
      factor = res.register(self.factor())
      if res.error: return res
      return res.success(UnaryOpNode(tok, factor))

    return self.power()

  def power(self):
    return self.bin_op(self.call, (TT_POW, ), self.factor)

  def call(self):
    res = ParseResult()
    atom = res.register(self.atom())
    if res.error: return res

    if self.current_tok.type == TT_LPAREN:
      res.register_advancement()
      self.advance()
      arg_nodes = []

      if self.current_tok.type == TT_RPAREN:
        res.register_advancement()
        self.advance()
      else:
        arg_nodes.append(res.register(self.expr()))
        if res.error:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected ')', 'var', 'if', 'for', 'while', 'fun', int, float, identifier, '+', '-', '(', '[' or 'not'"
          ))

        while self.current_tok.type == TT_COMMA:
          res.register_advancement()
          self.advance()

          arg_nodes.append(res.register(self.expr()))
          if res.error: return res

        if self.current_tok.type != TT_RPAREN:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected ',' or ')'"
          ))

        res.register_advancement()
        self.advance()
      return res.success(CallNode(atom, arg_nodes))
    return res.success(atom)

  def atom(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TT_INT, TT_FLOAT):
      res.register_advancement()
      self.advance()
      return res.success(NumberNode(tok))

    elif tok.type == TT_STRING:
      res.register_advancement()
      self.advance()
      return res.success(StringNode(tok))

    elif tok.type == TT_IDENTIFIER:
      res.register_advancement()
      self.advance()
      return res.success(VarAccessNode(tok))

    elif tok.type == TT_LPAREN:
      res.register_advancement()
      self.advance()
      expr = res.register(self.expr())
      if res.error: return res
      if self.current_tok.type == TT_RPAREN:
        res.register_advancement()
        self.advance()
        return res.success(expr)
      else:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ')'"
        ))

    elif tok.type == TT_LSQUARE:
      list_expr = res.register(self.list_expr())
      if res.error: return res
      return res.success(list_expr)
    

    elif tok.matches(TT_KEYWORD, 'if'):
      if_expr = res.register(self.if_expr())
      if res.error: return res
      return res.success(if_expr)
    
    elif tok.matches(TT_KEYWORD, 'try'):
      try_expr = res.register(self.try_expr())
      if res.error: return res
      return res.success(try_expr)

    elif tok.matches(TT_KEYWORD, 'throw'):
      throw_expr = res.register(self.throw_expr())
      if res.error: return res
      return res.success(throw_expr)

    elif tok.matches(TT_KEYWORD, 'import'):
      import_expr = res.register(self.import_expr())
      if res.error: return res
      return res.success(import_expr)
    
    elif tok.matches(TT_KEYWORD, 'assert'):
      assert_expr = res.register(self.assert_expr())
      if res.error: return res
      return res.success(assert_expr)

    elif tok.matches(TT_KEYWORD, 'for'):
      for_expr = res.register(self.for_expr())
      if res.error: return res
      return res.success(for_expr)

    elif tok.matches(TT_KEYWORD, 'while'):
      while_expr = res.register(self.while_expr())
      if res.error: return res
      return res.success(while_expr)

    elif tok.matches(TT_KEYWORD, 'fun'):
      func_def = res.register(self.func_def())
      if res.error: return res
      return res.success(func_def)

    return res.failure(InvalidSyntaxError(
      tok.pos_start, tok.pos_end,
      "Expected int, float, identifier, '+', '-', '(', '[', '{', if', 'for', 'while', 'fun'"
    ))

  def list_expr(self):
    res = ParseResult()
    element_nodes = []
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.type != TT_LSQUARE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '['"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_RSQUARE:
      res.register_advancement()
      self.advance()
    else:
      element_nodes.append(res.register(self.expr()))
      if res.error:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ']', 'var', 'if', 'for', 'while', 'fun', int, float, identifier, '+', '-', '(', '[' or 'not'"
        ))

      while self.current_tok.type == TT_COMMA:
        res.register_advancement()
        self.advance()

        element_nodes.append(res.register(self.expr()))
        if res.error: return res

      if self.current_tok.type != TT_RSQUARE:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected ',' or ']'"
        ))

      res.register_advancement()
      self.advance()

    return res.success(ListNode(
      element_nodes,
      pos_start,
      self.current_tok.pos_end.copy()
    ))

  def throw_expr(self):
    res = ParseResult()
    if not self.current_tok.matches(TT_KEYWORD, 'throw'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'throw'"
      ))
    
    res.register_advancement()
    self.advance()

    if self.current_tok.type != TT_STRING:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected string"
      ))
    
    details = self.current_tok
    res.register_advancement()
    self.advance()
    return res.success(ThrowNode(details, self.current_tok.pos_start, self.current_tok.pos_end))

  #############################
  def try_expr(self):
    res = ParseResult()
    all_cases = res.register(self.try_expr_cases('try'))
    if res.error: return res
    cases, finally_case = all_cases
    return res.success(TryNode(cases, finally_case))

  def try_expr_b(self):
    return self.try_expr_cases('catch')
    
  def try_expr_c(self):
    res = ParseResult()
    finally_case = None

    if self.current_tok.matches(TT_KEYWORD, 'finally'):
      res.register_advancement()
      self.advance()

      if self.current_tok.type != TT_LBRACE:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos, self.current_tok.pos_end,
          "Expected '{'"
        ))
      res.register_advancement()
      self.advance()

      statements = res.register(self.statements())
      if res.error: return res
      finally_case = (statements,)

      if self.current_tok.type == TT_RBRACE:
        res.register_advancement()
        self.advance()
      else:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected '}'"
        ))

    return res.success(finally_case)

  def try_expr_b_or_c(self):
    res = ParseResult()
    cases, finally_case = [], None

    if self.current_tok.matches(TT_KEYWORD, 'catch'):
      all_cases = res.register(self.try_expr_b())
      if res.error: return res
      cases, finally_case = all_cases
    else:
      finally_case = res.register(self.try_expr_c())
      if res.error: return res
    
    return res.success((cases, finally_case))

  def try_expr_cases(self, case_keyword):
    res = ParseResult()
    cases = []
    finally_case = None

    if not self.current_tok.matches(TT_KEYWORD, case_keyword):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '{case_keyword}'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type != TT_LBRACE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected '{'"
      ))

    res.register_advancement()
    self.advance()

    statements = res.register(self.statements())
    if res.error: return res
    cases.append((statements,))

    if self.current_tok.type == TT_RBRACE:
      res.register_advancement()
      self.advance()

    all_cases = res.register(self.try_expr_b_or_c())
    if res.error: return res
    new_cases, finally_case = all_cases
    cases.extend(new_cases)
    return res.success((cases, finally_case))
  #############################

  def if_expr(self):
    res = ParseResult()
    all_cases = res.register(self.if_expr_cases('if'))
    if res.error: return res
    cases, else_case = all_cases
    return res.success(IfNode(cases, else_case))

  def if_expr_b(self):
    return self.if_expr_cases('elsif')
    
  def if_expr_c(self):
    res = ParseResult()
    else_case = None

    if self.current_tok.matches(TT_KEYWORD, 'else'):
      res.register_advancement()
      self.advance()

      if self.current_tok.type == TT_LBRACE:
        res.register_advancement()
        self.advance()

        statements = res.register(self.statements())
        if res.error: return res
        else_case = (statements, True)

        if self.current_tok.type == TT_RBRACE:
          res.register_advancement()
          self.advance()
        else:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected '}'"
          ))
      else:
        expr = res.register(self.statement())
        if res.error: return res
        else_case = (expr, False)

    return res.success(else_case)

  def if_expr_b_or_c(self):
    res = ParseResult()
    cases, else_case = [], None

    if self.current_tok.matches(TT_KEYWORD, 'elsif'):
      all_cases = res.register(self.if_expr_b())
      if res.error: return res
      cases, else_case = all_cases
    else:
      else_case = res.register(self.if_expr_c())
      if res.error: return res
    
    return res.success((cases, else_case))

  def if_expr_cases(self, case_keyword):
    res = ParseResult()
    cases = []
    else_case = None

    if not self.current_tok.matches(TT_KEYWORD, case_keyword):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '{case_keyword}'"
      ))

    res.register_advancement()
    self.advance()

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.type == TT_LBRACE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected '{'"
      ))

    res.register_advancement()
    self.advance()

    statements = res.register(self.statements())
    if res.error: return res
    cases.append((condition, statements, True))

    if self.current_tok.type == TT_RBRACE:
      res.register_advancement()
      self.advance()
    # else:
    all_cases = res.register(self.if_expr_b_or_c())
    if res.error: return res
    new_cases, else_case = all_cases
    cases.extend(new_cases)

    return res.success((cases, else_case))

  def assert_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'assert'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'assert'"
      ))
    
    res.register_advancement()
    self.advance()

    condition = res.register(self.expr())
    if res.error: return res
    
    res.register_advancement()
    self.advance()
    return res.success(AssertNode(condition, self.current_tok.pos_start, self.current_tok.pos_end))

  def import_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'import'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'import'"
      ))
    
    res.register_advancement()
    self.advance()

    if self.current_tok.type != TT_STRING:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected string"
      ))
    
    file_name = self.current_tok

    res.register_advancement()
    self.advance()
    return res.success(ImportNode(file_name))

  def for_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'for'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'for'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type != TT_IDENTIFIER:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected identifier"
      ))

    var_name = self.current_tok
    res.register_advancement()
    self.advance()

    if not self.current_tok.matches(TT_KEYWORD, 'from'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'from'"
      ))
    
    res.register_advancement()
    self.advance()

    start_value = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'to'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'to'"
      ))
    
    res.register_advancement()
    self.advance()

    end_value = res.register(self.expr())
    if res.error: return res

    if self.current_tok.matches(TT_KEYWORD, 'by'):
      res.register_advancement()
      self.advance()

      step_value = res.register(self.expr())
      if res.error: return res
    else:
      step_value = None

    if not self.current_tok.type == TT_LBRACE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected '{'"
      ))

    res.register_advancement()
    self.advance()

    body = res.register(self.statements())
    if res.error: return res

    if not self.current_tok.type == TT_RBRACE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected '}'"
      ))

    res.register_advancement()
    self.advance()

    return res.success(ForNode(var_name, start_value, end_value, step_value, body, True))

  def while_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'while'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'while'"
      ))

    res.register_advancement()
    self.advance()

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.type == TT_LBRACE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected '{'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

      body = res.register(self.statements())
      if res.error: return res

      if not self.current_tok.type == TT_RBRACE:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected '}'"
        ))

      res.register_advancement()
      self.advance()

      return res.success(WhileNode(condition, body, True))
    
    body = res.register(self.statement())
    if res.error: return res

    return res.success(WhileNode(condition, body, False))

  def func_def(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'fun'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'fun'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_IDENTIFIER:
      var_name_tok = self.current_tok
      res.register_advancement()
      self.advance()
      if self.current_tok.type != TT_LPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected '('"
        ))
    else:
      var_name_tok = None
      if self.current_tok.type != TT_LPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected identifier or '('"
        ))
    
    res.register_advancement()
    self.advance()
    arg_name_toks = []

    if self.current_tok.type == TT_IDENTIFIER:
      arg_name_toks.append(self.current_tok)
      res.register_advancement()
      self.advance()
      
      while self.current_tok.type == TT_COMMA:
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_IDENTIFIER:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected identifier"
          ))

        arg_name_toks.append(self.current_tok)
        res.register_advancement()
        self.advance()
      
      if self.current_tok.type != TT_RPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected ',' or ')'"
        ))
    else:
      if self.current_tok.type != TT_RPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected identifier or ')'"
        ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_ARROW:
      res.register_advancement()
      self.advance()

      body = res.register(self.expr())
      if res.error: return res

      return res.success(FuncDefNode(
        var_name_tok,
        arg_name_toks,
        body,
        True
      ))
    
    if self.current_tok.type != TT_LBRACE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected '->' or '{'"
      ))

    res.register_advancement()
    self.advance()

    body = res.register(self.statements())
    if res.error: return res

    if not self.current_tok.type == TT_RBRACE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected '}'"
      ))

    res.register_advancement()
    self.advance()
    
    return res.success(FuncDefNode(
      var_name_tok,
      arg_name_toks,
      body,
      False
    ))

  ###################################

  def bin_op(self, func_a, ops, func_b=None):
    if func_b == None:
      func_b = func_a
    
    res = ParseResult()
    left = res.register(func_a())
    if res.error: return res

    while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()
      right = res.register(func_b())
      if res.error: return res
      left = BinOpNode(left, op_tok, right)

    return res.success(left)

#######################################
# Runtime Result
#######################################

class RTResult:
  def __init__(self):
    self.reset()

  def reset(self):
    self.value = None
    self.error = None
    self.func_return_value = None
    self.loop_should_continue = False
    self.loop_should_break = False

  def register(self, res):
    self.error = res.error
    self.func_return_value = res.func_return_value
    self.loop_should_continue = res.loop_should_continue
    self.loop_should_break = res.loop_should_break
    return res.value

  def success(self, value):
    self.reset()
    self.value = value
    return self

  def success_return(self, value):
    self.reset()
    self.func_return_value = value
    return self
  
  def success_continue(self):
    self.reset()
    self.loop_should_continue = True
    return self

  def success_break(self):
    self.reset()
    self.loop_should_break = True
    return self

  def failure(self, error):
    self.reset()
    self.error = error
    return self

  def should_return(self):
    return (
      self.error or
      self.func_return_value or
      self.loop_should_continue or
      self.loop_should_break
    )

#######################################
# Values
#######################################

class Value:
  def __init__(self):
    self.set_pos()
    self.set_context()

  def set_pos(self, pos_start=None, pos_end=None):
    self.pos_start = pos_start
    self.pos_end = pos_end
    return self

  def set_context(self, context=None):
    self.context = context
    return self

  def lshift_by(self, other):
    return None, self.illegal_operation(other)

  def rshift_by(self, other):
    return None, self.illegal_operation(other)

  def added_to(self, other):
    return None, self.illegal_operation(other)

  def subbed_by(self, other):
    return None, self.illegal_operation(other)

  def multed_by(self, other):
    return None, self.illegal_operation(other)

  def dived_by(self, other):
    return None, self.illegal_operation(other)

  def powed_by(self, other):
    return None, self.illegal_operation(other)

  def get_comparsion_eq(self, other):
    return None, self.illegal_operation(other)

  def get_comparsion_ne(self, other):
    return None, self.illegal_operation(other)

  def get_comparsion_lt(self, other):
    return None, self.illegal_operation(other)

  def get_comparsion_gt(self, other):
    return None, self.illegal_operation(other)

  def get_comparsion_lte(self, other):
    return None, self.illegal_operation(other)

  def get_comparsion_gte(self, other):
    return None, self.illegal_operation(other)

  def anded_by(self, other):
    return None, self.illegal_operation(other)

  def ored_by(self, other):
    return None, self.illegal_operation(other)

  def notted(self, other):
    return None, self.illegal_operation(other)

  def execute(self, args):
    return RTResult().failure(self.illegal_operation())

  def copy(self):
    raise Exception('No copy method defined')

  def is_true(self):
    return False

  def illegal_operation(self, other=None):
    if not other: other = self
    return Ktplus_DtypeError(
      self.pos_start, other.pos_end,
      'Illegal operation',
      self.context
    )

class BaseStream(Value):
  pass

class ostream(BaseStream):
  def __init__(self, name):
    super().__init__()
    self.name = name

  def lshift_by(self, other):
    print(other, end='')
    return self.copy(), None

  def copy(self):
    copy = ostream(self.name)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __repr__(self):
    return f"<ostream {self.name}>"

class istream(BaseStream):
  def __init__(self, name):
    super().__init__()
    self.name = name
    
  def copy(self):
    copy = istream(self.name)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy
  
  def rshift_by(self, other):
    other_value = self.context.symbol_table.get(other)
    self.context.symbol_table.set(str(other), String(input()))
    return self.copy(), None
  
  def __repr__(self):
    return f"<istream {self.name}>"

ostream.kout = ostream("kout")
istream.kin  = istream("kin")

class Number(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value
  
  def lshift_by(self, other):
    if isinstance(other, Number):
      return Number(self.value << other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def rshift_by(self, other):
    if isinstance(other, Number):
      return Number(self.value >> other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)


  def added_to(self, other):
    if isinstance(other, Number):
      return Number(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def subbed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value - other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      if other.value == 0:
        return None, Ktplus_ZeroDivError(
          other.pos_start, other.pos_end,
          'Division by zero',
          self.context
        )

      return Number(self.value / other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def powed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value ** other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparsion_eq(self, other):
    if isinstance(other, Number):
      return Boolean(int(self.value == other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparsion_ne(self, other):
    if isinstance(other, Number):
      return Boolean(int(self.value != other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparsion_lt(self, other):
    if isinstance(other, Number):
      return Boolean(int(self.value < other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparsion_gt(self, other):
    if isinstance(other, Number):
      return Boolean(int(self.value > other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparsion_lte(self, other):
    if isinstance(other, Number):
      return Boolean(int(self.value <= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparsion_gte(self, other):
    if isinstance(other, Number):
      return Boolean(int(self.value >= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def anded_by(self, other):
    if isinstance(other, Number):
      return Boolean(int(self.value and other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def ored_by(self, other):
    if isinstance(other, Number):
      return Boolean(int(self.value or other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def notted(self):
    return Boolean(1 if self.value == 0 else 0).set_context(self.context), None

  def copy(self):
    copy = Number(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != 0

  def __int__(self):
    return int(self.value)
  
  def __float__(self):
    return float(self.value)

  def __str__(self):
    return str(self.value)
  
  def __repr__(self):
    return str(self.value)

class Boolean(Value):
  def __init__(self, value):
    super().__init__()
    if value: self.value = True
    else: self.value = False
  
  def copy(self):
    copy = Boolean(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy
  
  def is_true(self):
    return self.value
  
  def __repr__(self):
    return "true" if self.value else "false"

class NullType(Value):
  def __init__(self):
    super().__init__()
    self.value = 0

  def added_to(self, other):
    return NullType().set_context(self.context), None

  def subbed_by(self, other):
    return NullType().set_context(self.context), None

  def multed_by(self, other):
    return NullType().set_context(self.context), None

  def dived_by(self, other):
    return NullType().set_context(self.context), None

  def powed_by(self, other):
    return NullType().set_context(self.context), None

  def get_comparsion_eq(self, other):
    return NullType().set_context(self.context), None

  def get_comparsion_ne(self, other):
    return NullType().set_context(self.context), None

  def get_comparsion_lt(self, other):
    return NullType().set_context(self.context), None

  def get_comparsion_gt(self, other):
    return NullType().set_context(self.context), None

  def get_comparsion_lte(self, other):
    return NullType().set_context(self.context), None

  def get_comparsion_gte(self, other):
    return NullType().set_context(self.context), None

  def anded_by(self, other):
    return NullType().set_context(self.context), None

  def ored_by(self, other):
    return NullType().set_context(self.context), None

  def notted(self):
    return NullType().set_context(self.context), None

  def copy(self):
    copy = NullType()
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return 0

  def __int__(self):
    return 0

  def __str__(self):
    return "null"
  
  def __repr__(self):
    return "null"

Number.null = NullType()
Number.false = Boolean(0)
Number.true = Boolean(1)
Number.math_pi = Number(math.pi)

class Represent(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

  def added_to(self, other):
    if isinstance(other, String):
      return Represent(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return Represent(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def is_true(self):
    return len(self.value) > 0

  def copy(self):
    copy = Represent(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __str__(self):
    return f'{self.value}'

  def __repr__(self):
    return f'{self.value}'

class String(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

  def added_to(self, other):
    if isinstance(other, String):
      return String(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return String(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def is_true(self):
    return len(self.value) > 0

  def copy(self):
    copy = String(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __str__(self):
    return self.value

  def __repr__(self):
    if isinstance(self.value, str):
      return f'"%s"' % self.value.replace('"', '\\"')

class List(Value):
  def __init__(self, elements):
    super().__init__()
    self.elements = elements

  def added_to(self, other):
    new_list = self.copy()
    new_list.elements.append(other)
    return new_list, None

  def subbed_by(self, other):
    if isinstance(other, Number):
      new_list = self.copy()
      try:
        new_list.elements.pop(other.value)
        return new_list, None
      except:
        return None, Ktplus_IdxError(
          other.pos_start, other.pos_end,
          'Element at this index could not be removed from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, List):
      new_list = self.copy()
      new_list.elements.extend(other.elements)
      return new_list, None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      try:
        return self.elements[other.value], None
      except:
        return None, Ktplus_IdxError(
          other.pos_start, other.pos_end,
          'Element at this index could not be retrieved from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)
  
  def copy(self):
    copy = List(self.elements)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __iter__(self):
    return iter(self.elements)

  def __str__(self):
    return f'[{", ".join([str(x) for x in self.elements])}]'

  def __repr__(self):
    return f'[{", ".join([repr(x) for x in self.elements])}]'


class BaseFunction(Value):
  def __init__(self, name):
    super().__init__()
    self.name = name or "<lambda>"

  def generate_new_context(self):
    new_context = Context(self.name, self.context, self.pos_start)
    new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
    return new_context

  def check_args(self, arg_names, args):
    res = RTResult()

    if len(args) != len(arg_names):
      if len(arg_names) == 1:
        return res.failure(Ktplus_UsageError(
          self.pos_start, self.pos_end,
          f"{self} takes {len(arg_names)} argument but {len(args)} were given",
          self.context
        ))
      else:
        return res.failure(Ktplus_UsageError(
          self.pos_start, self.pos_end,
          f"{self} takes {len(arg_names)} arguments but {len(args)} were given",
          self.context
        ))

    return res.success(None)

  def populate_args(self, arg_names, args, exec_ctx):
    for i in range(len(args)):
      arg_name = arg_names[i]
      arg_value = args[i]
      arg_value.set_context(exec_ctx)
      exec_ctx.symbol_table.set(arg_name, arg_value)

  def check_and_populate_args(self, arg_names, args, exec_ctx):
    res = RTResult()
    res.register(self.check_args(arg_names, args))
    if res.should_return(): return res
    self.populate_args(arg_names, args, exec_ctx)
    return res.success(None)

class Function(BaseFunction):
  def __init__(self, name, body_node, arg_names, should_auto_return):
    super().__init__(name)
    self.body_node = body_node
    self.arg_names = arg_names
    self.should_auto_return = should_auto_return

  def execute(self, args):
    res = RTResult()
    interpreter = Interpreter()
    exec_ctx = self.generate_new_context()

    res.register(self.check_and_populate_args(self.arg_names, args, exec_ctx))
    if res.should_return(): return res

    value = res.register(interpreter.visit(self.body_node, exec_ctx))
    if res.should_return() and res.func_return_value == None: return res

    ret_value = (value if self.should_auto_return else None) or res.func_return_value or Number.null
    return res.success(ret_value)

  def copy(self):
    copy = Function(self.name, self.body_node, self.arg_names, self.should_auto_return)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __repr__(self):
    return f"<function {self.name}>" if self.name != "<lambda>" else f"<lambda function at {hex(id(self))}>"

class BuiltInFunction(BaseFunction):
  def __init__(self, name):
    super().__init__(name)

  def execute(self, args):
    res = RTResult()
    exec_ctx = self.generate_new_context()

    method_name = f'execute_{self.name}'
    method = getattr(self, method_name, self.no_visit_method)

    res.register(self.check_and_populate_args(method.arg_names, args, exec_ctx))
    if res.should_return(): return res

    return_value = res.register(method(exec_ctx))
    if res.should_return(): return res
    return res.success(return_value)
  
  def no_visit_method(self, node, context):
    raise Exception(f'No execute_{self.name} method defined')

  def copy(self):
    copy = BuiltInFunction(self.name)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __repr__(self):
    return f"<built-in function {self.name}>"

  #####################################

  def execute_indexof(self, exec_ctx):
    iterab = exec_ctx.symbol_table.get("iterable")
    index = exec_ctx.symbol_table.get("index")
    try:
      return RTResult().success(iterab.elements[int(index)])
    except Exception as e:
      return RTResult().failure(Ktplus_IdxError(
        self.pos_start, self.pos_end,
        "invalid index operation",
        exec_ctx
      ))
    
  execute_indexof.arg_names = ["iterable", "index"]

  def execute_typeof(self, exec_ctx):
    return RTResult().success(Represent(type(exec_ctx.symbol_table.get("obj")).__name__))
  execute_typeof.arg_names = ["obj"]

  def execute_mod(self, exec_ctx):
    return RTResult().success(Number(int(exec_ctx.symbol_table.get('num1')) % int(exec_ctx.symbol_table.get('num2'))))
  execute_mod.arg_names = ['num1', 'num2']

  def execute_rand(self, exec_ctx):
    return RTResult().success(Number(int(str(random.random()).partition(".")[2])))
  execute_rand.arg_names = []
  
  def execute_int(self, exec_ctx):
    try:
      num = int(str(exec_ctx.symbol_table.get('num')).partition(".")[0])
    except ValueError:
      return RTResult().failure(Ktplus_ValueError(
        self.pos_start, self.pos_end,
        "invalid literal",
        exec_ctx
      ))
    return RTResult().success(Number(int(num)))
  execute_int.arg_names = ['num']

  def execute_format(self, exec_ctx):
    try:
      result = String(str(exec_ctx.symbol_table.get('value')) % tuple(eval(repr(exec_ctx.symbol_table.get('formats')))))
    except:
      return RTResult().failure(Ktplus_ValueError(
        self.pos_start, self.pos_end,
        "Error when formatting string",
        exec_ctx
      ))
    return RTResult().success(result)
  execute_format.arg_names = ['value', 'formats']

  def execute_printf(self, exec_ctx):
    try:
      print(str(exec_ctx.symbol_table.get('value')) % tuple(exec_ctx.symbol_table.get('formats')))
    except:
      return RTResult().failure(Ktplus_ValueError(
        self.pos_start, self.pos_end,
        "Error when formatting string",
        exec_ctx
      ))
    return RTResult().success(Number.null)
  execute_printf.arg_names = ['value', 'formats']

  def execute_print(self, exec_ctx):
    print(str(exec_ctx.symbol_table.get('value')))
    return RTResult().success(Number.null)
  execute_print.arg_names = ['value']
  
  def execute_exit(self, exec_ctx):
    status = exec_ctx.symbol_table.get('status')
    if not isinstance(status, Number):
      return RTResult.failure(Ktplus_ValueError(
        self.pos_start, self.pos_end,
        "First argument must be a number",
        exec_ctx
      ))
    os._exit(int(status))
    return RTResult().success(Number.null)
  execute_exit.arg_names = ['status']

  def execute_str(self, exec_ctx):
    return RTResult().success(String(str(exec_ctx.symbol_table.get('value'))))
  execute_str.arg_names = ['value']
  
  def execute_input(self, exec_ctx):
    text = input(str(exec_ctx.symbol_table.get('prompt')))
    return RTResult().success(String(text))
  execute_input.arg_names = ['prompt']

  def execute_clear(self, exec_ctx):
    os.system('cls' if os.name == 'nt' else 'cls') 
    return RTResult().success(Number.null)
  execute_clear.arg_names = []

  def execute_vars(self, exec_ctx):
    return RTResult().success(Represent(exec_ctx.symbol_table.symbols))
  execute_vars.arg_names = []

  def execute_is_number(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), Number)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_number.arg_names = ["value"]

  def execute_is_string(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), String)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_string.arg_names = ["value"]

  def execute_is_list(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), List)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_list.arg_names = ["value"]

  def execute_is_function(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), BaseFunction)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_function.arg_names = ["value"]

  def execute_append(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    value = exec_ctx.symbol_table.get("value")

    if not isinstance(list_, List):
      return RTResult().failure(Ktplus_DtypeError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    list_.elements.append(value)
    return RTResult().success(Number.null)
  execute_append.arg_names = ["list", "value"]

  def execute_pop(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    index = exec_ctx.symbol_table.get("index")

    if not isinstance(list_, List):
      return RTResult().failure(Ktplus_DtypeError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    if not isinstance(index, Number):
      return RTResult().failure(Ktplus_DtypeError(
        self.pos_start, self.pos_end,
        "Second argument must be number",
        exec_ctx
      ))

    try:
      element = list_.elements.pop(index.value)
    except:
      return RTResult().failure(Ktplus_IdxError(
        self.pos_start, self.pos_end,
        'Element at this index could not be removed from list because index is out of bounds',
        exec_ctx
      ))
    return RTResult().success(element)
  execute_pop.arg_names = ["list", "index"]

  def execute_extend(self, exec_ctx):
    listA = exec_ctx.symbol_table.get("listA")
    listB = exec_ctx.symbol_table.get("listB")

    if not isinstance(listA, List):
      return RTResult().failure(Ktplus_DtypeError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    if not isinstance(listB, List):
      return RTResult().failure(Ktplus_DtypeError(
        self.pos_start, self.pos_end,
        "Second argument must be list",
        exec_ctx
      ))

    listA.elements.extend(listB.elements)
    return RTResult().success(Number.null)
  execute_extend.arg_names = ["listA", "listB"]

  def execute_len(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")

    if not (isinstance(list_, List)):
      return RTResult().failure(Ktplus_DtypeError(
        self.pos_start, self.pos_end,
        "Argument must be list",
        exec_ctx
      ))
    
    return RTResult().success(Number(len(list_.elements)))
  execute_len.arg_names = ["list"]


  def execute_include(self, exec_ctx):
    fn = exec_ctx.symbol_table.get("fn")
    exec(f"import stdlib.kinclude.{fn} as {fn}")
    exec("setattr(BuiltInFunction, 'execute_{fn}', {fn}.main)".format(fn=fn))
    exec("setattr(BuiltInFunction, '{fn}', BuiltInFunction('{fn}'))".format(fn=fn))
    exec("global_symbol_table.set('{fn}',BuiltInFunction.{fn})".format(fn=fn))
    return RTResult().success(Number.null)
  execute_include.arg_names = ["fn"]
  
  def execute_import_module(self, exec_ctx):
    fn = exec_ctx.symbol_table.get("fn")

    if not isinstance(fn, String):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be string",
        exec_ctx
      ))

    fn = fn.value

    try:
      with open("./stdlib/"+fn+'.ktplus', "r") as f:
        script = f.read()
    except Exception as e:
      return RTResult().failure(Ktplus_ValueError(
        self.pos_start, self.pos_end,
        f"Failed to load library \"{fn}\"\n" + str(e),
        exec_ctx
      ))

    _, error = run(fn, script)
    
    if error:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        error.as_string(),
        exec_ctx,
      ))

    return RTResult().success(Number.null)
  execute_import_module.arg_names = ["fn"]

BuiltInFunction.print       = BuiltInFunction("print")
BuiltInFunction.str         = BuiltInFunction("str")
BuiltInFunction.int         = BuiltInFunction("int")
BuiltInFunction.input       = BuiltInFunction("input")
BuiltInFunction.clear       = BuiltInFunction("clear")
BuiltInFunction.is_number   = BuiltInFunction("is_number")
BuiltInFunction.is_string   = BuiltInFunction("is_string")
BuiltInFunction.is_list     = BuiltInFunction("is_list")
BuiltInFunction.is_function = BuiltInFunction("is_function")
BuiltInFunction.append      = BuiltInFunction("append")
BuiltInFunction.pop         = BuiltInFunction("pop")
BuiltInFunction.extend      = BuiltInFunction("extend")
BuiltInFunction.len					= BuiltInFunction("len")
BuiltInFunction.imp					= BuiltInFunction("import_module")
BuiltInFunction.printf      = BuiltInFunction("printf")
BuiltInFunction.format      = BuiltInFunction("format")
BuiltInFunction.exit        = BuiltInFunction("exit")
BuiltInFunction.rand        = BuiltInFunction("rand")
BuiltInFunction.mod         = BuiltInFunction("mod")
BuiltInFunction.throw       = BuiltInFunction("throw")
BuiltInFunction.typeof      = BuiltInFunction("typeof")
BuiltInFunction.vars        = BuiltInFunction("vars")
BuiltInFunction.include     = BuiltInFunction("include")
BuiltInFunction.indexof     = BuiltInFunction("indexof")

#######################################
# Context
#######################################

class Context:
  def __init__(self, display_name, parent=None, parent_entry_pos=None):
    self.display_name = display_name
    self.parent = parent
    self.parent_entry_pos = parent_entry_pos
    self.symbol_table = None

#######################################
# Symbol Table
#######################################

class SymbolTable:
  def __init__(self, parent=None):
    self.symbols = {}
    self.parent = parent

  def get(self, name):
    value = self.symbols.get(name, None)
    if value == None and self.parent:
      return self.parent.get(name)
    return value

  def set(self, name, value):
    self.symbols[name] = value

  def remove(self, name):
    del self.symbols[name]

#######################################
# Interpreter
#######################################

class Interpreter:
  def visit(self, node, context):
    method_name = f'visit_{type(node).__name__}'
    method = getattr(self, method_name, self.no_visit_method)
    return method(node, context)

  def no_visit_method(self, node, context):
    raise Exception(f'No visit_{type(node).__name__} method defined')

  ###################################

  def visit_NumberNode(self, node, context):
    return RTResult().success(
      Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_StringNode(self, node, context):
    return RTResult().success(
      String(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_ListNode(self, node, context):
    res = RTResult()
    elements = []

    for element_node in node.element_nodes:
      elements.append(res.register(self.visit(element_node, context)))
      if res.should_return(): return res

    return res.success(
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )


  def visit_ThrowNode(self, node, context):
    res = RTResult()
    details = node.details_tok
    return res.failure(RTError(
      node.pos_start, node.pos_end,
      details.value,
      context
    ))

  def visit_VarAccessNode(self, node, context):
    res = RTResult()
    var_name = node.var_name_tok.value
    value = context.symbol_table.get(var_name)

    if not value:
      return res.failure(Ktplus_NameError(
        node.pos_start, node.pos_end,
        f"'{var_name}' is not defined",
        context
      ))

    value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
    return res.success(value)

  def visit_VarAssignNode(self, node, context):
    res = RTResult()
    var_name = node.var_name_tok.value
    value = res.register(self.visit(node.value_node, context))
    if res.should_return(): return res

    context.symbol_table.set(var_name, value)
    return res.success(Number.null)

  def visit_BinOpNode(self, node, context):
    res = RTResult()
    left = res.register(self.visit(node.left_node, context))
    if res.should_return(): return res
    right = res.register(self.visit(node.right_node, context))
    if res.should_return(): return res

    if node.op_tok.type == TT_LSH:
      result, error = left.lshift_by(right)
    elif node.op_tok.type == TT_RSH:
      if type(left) == istream: 
        if not hasattr(node.right_node, 'var_name_tok'):
          error = Ktplus_DtypeError(
            node.pos_start, node.pos_end,
            f"Expected variable name after istream",
            context
          )
          return res.failure(error)
        result, error = left.rshift_by(node.right_node.var_name_tok.value)
      else:
        result, error = left.rshift_by(right)
    elif node.op_tok.type == TT_PLUS:
      result, error = left.added_to(right)
    elif node.op_tok.type == TT_MINUS:
      result, error = left.subbed_by(right)
    elif node.op_tok.type == TT_MUL:
      result, error = left.multed_by(right)
    elif node.op_tok.type == TT_DIV:
      result, error = left.dived_by(right)
    elif node.op_tok.type == TT_POW:
      result, error = left.powed_by(right)
    elif node.op_tok.type == TT_EE:
      result, error = left.get_comparsion_eq(right)
    elif node.op_tok.type == TT_NE:
      result, error = left.get_comparsion_ne(right)
    elif node.op_tok.type == TT_LT:
      result, error = left.get_comparsion_lt(right)
    elif node.op_tok.type == TT_GT:
      result, error = left.get_comparsion_gt(right)
    elif node.op_tok.type == TT_LTE:
      result, error = left.get_comparsion_lte(right)
    elif node.op_tok.type == TT_GTE:
      result, error = left.get_comparsion_gte(right)
    elif node.op_tok.matches(TT_KEYWORD, 'and'):
      result, error = left.anded_by(right)
    elif node.op_tok.matches(TT_KEYWORD, 'or'):
      result, error = left.ored_by(right)

    if error:
      return res.failure(error)
    else:
      return res.success(result.set_pos(node.pos_start, node.pos_end))

  def visit_UnaryOpNode(self, node, context):
    res = RTResult()
    number = res.register(self.visit(node.node, context))
    if res.should_return(): return res

    error = None

    if node.op_tok.type == TT_MINUS:
      number, error = number.multed_by(Number(-1))
    elif node.op_tok.matches(TT_KEYWORD, 'not'):
      number, error = number.notted()

    if error:
      return res.failure(error)
    else:
      return res.success(number.set_pos(node.pos_start, node.pos_end))

  def visit_IfNode(self, node, context):
    res = RTResult()

    for condition, expr, should_return_null in node.cases:
      condition_value = res.register(self.visit(condition, context))
      if res.should_return(): return res

      if condition_value.is_true():
        expr_value = res.register(self.visit(expr, context))
        if res.should_return(): return res
        return res.success(Number.null if should_return_null else expr_value)

    if node.else_case:
      expr, should_return_null = node.else_case
      expr_value = res.register(self.visit(expr, context))
      if res.should_return(): return res
      return res.success(Number.null if should_return_null else expr_value)

    return res.success(Number.null)

  def visit_TryNode(self, node, context):
    res = RTResult()

    expr_value = res.register(self.visit(node.cases[0][0], context))
    if res.error and len(node.cases) > 1:
      expr_value = res.register(self.visit(node.cases[1][0], context))
      if res.should_return(): return res

    if node.finally_case:
      expr = node.finally_case
      expr_value = res.register(self.visit(expr[0], context))
      if res.should_return(): return res
      return res.success(Number.null)
    return res.success(Number.null)

  def visit_AssertNode(self, node, context):
    res = RTResult()
    expr_res = self.visit(node.condition_node, context)
    if expr_res.error: return res.failure(expr_res.error)
    expr = expr_res.value
    try:
      assert expr.is_true()
    except AssertionError:
      return RTResult().failure(Ktplus_AssertError(
        node.pos_start, node.pos_end,
        str(expr),
        context
      ))
    return res.success(Number.null)

  def visit_ForNode(self, node, context):
    res = RTResult()
    elements = []

    start_value = res.register(self.visit(node.start_value_node, context))
    if res.should_return(): return res

    end_value = res.register(self.visit(node.end_value_node, context))
    if res.should_return(): return res

    if node.step_value_node:
      step_value = res.register(self.visit(node.step_value_node, context))
      if res.should_return(): return res
    else:
      step_value = Number(1)

    i = start_value.value

    if step_value.value >= 0:
      condition = lambda: i <= end_value.value
    else:
      condition = lambda: i >= end_value.value
    
    while condition():
      context.symbol_table.set(node.var_name_tok.value, Number(i))
      i += step_value.value

      value = res.register(self.visit(node.body_node, context))
      if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res
      
      if res.loop_should_continue:
        continue
      
      if res.loop_should_break:
        break

      elements.append(value)

    return res.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_WhileNode(self, node, context):
    res = RTResult()
    elements = []

    while True:
      condition = res.register(self.visit(node.condition_node, context))
      if res.should_return(): return res

      if not condition.is_true():
        break

      value = res.register(self.visit(node.body_node, context))
      if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

      if res.loop_should_continue:
        continue
      
      if res.loop_should_break:
        break

      elements.append(value)

    return res.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_FuncDefNode(self, node, context):
    res = RTResult()

    func_name = node.var_name_tok.value if node.var_name_tok else None
    body_node = node.body_node
    arg_names = [arg_name.value for arg_name in node.arg_name_toks]
    func_value = Function(func_name, body_node, arg_names, node.should_auto_return).set_context(context).set_pos(node.pos_start, node.pos_end)
    
    if node.var_name_tok:
      context.symbol_table.set(func_name, func_value)

    return res.success(func_value)

  def visit_CallNode(self, node, context):
    res = RTResult()
    args = []

    value_to_call = res.register(self.visit(node.node_to_call, context))
    if res.should_return(): return res
    value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

    for arg_node in node.arg_nodes:
      args.append(res.register(self.visit(arg_node, context)))
      if res.should_return(): return res

    return_value = res.register(value_to_call.execute(args))
    if res.should_return(): return res
    return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
    return res.success(return_value)

  def visit_ReturnNode(self, node, context):
    res = RTResult()

    if node.node_to_return:
      value = res.register(self.visit(node.node_to_return, context))
      if res.should_return(): return res
    else:
      value = Number.null
    
    return res.success_return(value)
  
  def visit_ImportNode(self, node, context):
    res = RTResult()
    fn = node.tok

    fn = fn.value

    try:
      with open("./stdlib/"+fn+'.ktplus', "r") as f:
        script = f.read()
    except:
      return RTResult().failure(Ktplus_FileNotFoundError(
        node.pos_start, node.pos_end,
        f"Failed to load library '{fn}'",
        context
      ))

    _, error = run(f"<stdlib '{fn}'>", script)

    if error:
      return RTResult().failure(RTError(
        node.pos_start, node.pos_end,
        f"Failed to finish executing library '{fn}'\n" + error.as_string(),
        context
      ))

    return RTResult().success(Number.null)
    

  def visit_ContinueNode(self, node, context):
    return RTResult().success_continue()

  def visit_BreakNode(self, node, context):
    return RTResult().success_break()

#######################################
# Global Symbol Table Initialize
#######################################

global_symbol_table = SymbolTable()
global_symbol_table.set("null", Number.null)
global_symbol_table.set("false", Number.false)
global_symbol_table.set("true", Number.true)
global_symbol_table.set("pi", Number.math_pi)
global_symbol_table.set("print", BuiltInFunction.print)
global_symbol_table.set("input", BuiltInFunction.input)
global_symbol_table.set("clear", BuiltInFunction.clear)
global_symbol_table.set("cls", BuiltInFunction.clear)
global_symbol_table.set("is_number", BuiltInFunction.is_number)
global_symbol_table.set("is_string", BuiltInFunction.is_string)
global_symbol_table.set("is_list", BuiltInFunction.is_list)
global_symbol_table.set("is_function", BuiltInFunction.is_function)
global_symbol_table.set("append", BuiltInFunction.append)
global_symbol_table.set("pop", BuiltInFunction.pop)
global_symbol_table.set("extend", BuiltInFunction.extend)
global_symbol_table.set("len", BuiltInFunction.len)
global_symbol_table.set("import_module", BuiltInFunction.imp)
global_symbol_table.set("int", BuiltInFunction.int)
global_symbol_table.set("str", BuiltInFunction.str)
global_symbol_table.set("printf", BuiltInFunction.printf)
global_symbol_table.set("format", BuiltInFunction.format)
global_symbol_table.set("exit", BuiltInFunction.exit)
global_symbol_table.set("rand", BuiltInFunction.rand)
global_symbol_table.set("mod", BuiltInFunction.mod)
global_symbol_table.set("typeof", BuiltInFunction.typeof)
global_symbol_table.set("vars", BuiltInFunction.vars)
global_symbol_table.set("include", BuiltInFunction.include)
global_symbol_table.set("indexof", BuiltInFunction.indexof)
global_symbol_table.set("kin", istream.kin)
global_symbol_table.set("kout", ostream.kout)

#######################################
# Run
#######################################

def run(fn, text):
  # tokens
  scanner = Scanner(fn, text)
  tokens, error = scanner.make_tokens()
  if error: return None, error
  
  # AST
  parser = Parser(tokens)
  ast = parser.parse()
  if ast.error: return None, ast.error

  # Run
  interpreter = Interpreter()
  context = Context('<module>')
  context.symbol_table = global_symbol_table
  result = interpreter.visit(ast.node, context)

  return result.value, result.error